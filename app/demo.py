"""
EmotionScope Gradio Demo — Real conversation with emotion orb indicators.

The model generates REAL responses via model.generate(). Emotion probing
happens during the forward passes that generation itself performs — no
separate probe-only pass.

Architecture per user message:
  Step A: Probe user message → user_emotion_state
  Step B: Generate real response, hook first token → model_emotion_state
  Step C: Return clean text + JSON emotion states to orb renderer

Usage:
    uv run python app/demo.py
    uv run python app/demo.py --model google/gemma-2-2b-it --vectors results/vectors/google_gemma-2-2b-it.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import gradio as gr

from emotion_scope.config import CORE_EMOTIONS
from emotion_scope.extract import EmotionExtractor
from emotion_scope.models import load_model
from emotion_scope.probe import EmotionProbe
from emotion_scope.visualize import scores_to_orb_state

# ── Global state (loaded once at startup) ──
_probe: Optional[EmotionProbe] = None
_model = None
_tokenizer = None
_backend: str = ""
_info: dict = {}


def init_probe(model_name: str, vectors_path: str, device: str = "auto"):
    global _probe, _model, _tokenizer, _backend, _info

    saved = EmotionExtractor.load(vectors_path)
    vectors = saved["vectors"]
    emotions = saved.get("emotions", CORE_EMOTIONS)
    saved_info = saved["model_info"]

    _model, _tokenizer, _backend, _info = load_model(
        model_name=model_name, device=device, run_smoke_test=False)
    _info["probe_layer"] = saved.get("probe_layer_used", saved_info["probe_layer"])

    _probe = EmotionProbe(
        model=_model, tokenizer=_tokenizer, backend=_backend,
        model_info=_info, emotion_vectors=vectors, emotions_metadata=emotions)
    print(f"[demo] Probe ready: {model_name} @ layer {_info['probe_layer']}")


# =========================================================================
#  Step A: Probe user message
# =========================================================================

def probe_user_message(conversation: list[dict]) -> dict:
    """Run a forward pass on the conversation up to the user's message and
    extract the emotion state at the last content token."""
    if _probe is None:
        return {}
    dual = _probe.analyze_conversation(
        user_message=conversation[-1]["content"],
        system_prompt=None,
    )
    return scores_to_orb_state(dual.model_state.scores)


# =========================================================================
#  Step B: Generate real response + capture model emotion at first token
# =========================================================================

def generate_with_probe(conversation: list[dict]) -> tuple[str, dict]:
    """Generate a real response and capture the residual stream at the
    first generated token for the model's emotion state."""

    probe_layer = _info["probe_layer"]
    captured: dict = {}

    # Build chat prompt
    apply = getattr(_tokenizer, "apply_chat_template", None)
    if apply is not None:
        prompt = apply(conversation, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in conversation
        ) + "\nAssistant:"

    if _backend == "transformer_lens":
        input_ids = _model.to_tokens(prompt)
        input_len = input_ids.shape[1]

        # Hook into the TransformerLens block to capture first generated token.
        # TL's generate() calls forward() internally for each new token.
        hook_name = f"blocks.{probe_layer}.hook_resid_post"

        def capture_hook(activation, hook):
            if "first_token" not in captured:
                captured["first_token"] = activation[0, -1, :].detach().cpu().float()
            return activation

        # Use model.add_hook (TL's generate doesn't accept fwd_hooks kwarg)
        _model.add_hook(hook_name, capture_hook)
        try:
            output_ids = _model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                stop_at_eos=True,
            )
        finally:
            _model.reset_hooks()

        # Decode only the new tokens
        new_token_ids = output_ids[0, input_len:]
        response_text = _model.tokenizer.decode(
            new_token_ids, skip_special_tokens=True)

    else:
        # HuggingFace backend
        tokens = _tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(_model.device)
        input_len = input_ids.shape[1]
        attention_mask = tokens.get("attention_mask", torch.ones_like(input_ids)).to(_model.device)

        # Hook on the HF layer
        layers = _find_layers(_model)

        def hf_hook(_module, _input, output):
            if "first_token" not in captured:
                act = output[0] if isinstance(output, tuple) else output
                captured["first_token"] = act[0, -1, :].detach().cpu().float()

        handle = layers[probe_layer].register_forward_hook(hf_hook)
        try:
            output_ids = _model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        finally:
            handle.remove()

        new_token_ids = output_ids[0, input_len:]
        response_text = _tokenizer.decode(new_token_ids, skip_special_tokens=True)

    # Score the captured activation
    model_state = {}
    if "first_token" in captured and _probe is not None:
        emotion = _probe.analyze_activation(captured["first_token"])
        model_state = scores_to_orb_state(emotion.scores)

    return response_text.strip(), model_state


def _find_layers(model):
    for attr_path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in attr_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


# =========================================================================
#  Chat function — ties Steps A, B, C together
# =========================================================================

def chat_fn(message: str, history: list[dict]):
    """Process a user message: probe → generate → return clean text + states."""
    if _probe is None:
        return "Model not loaded.", history, "{}", "{}"

    # Build conversation history for the model
    conversation = list(history) + [{"role": "user", "content": message}]

    # Step A: Probe user message
    user_state = probe_user_message(conversation)

    # Step B: Generate real response + probe model state
    response_text, model_state = generate_with_probe(conversation)

    # Step C: Update history with clean text only
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response_text})

    # Combine both states for the orb
    orb_data = {
        "model": model_state,
        "user": user_state,
    }

    timeline_entry = {
        "user_color": user_state.get("color_hex", "#333"),
        "model_color": model_state.get("color_hex", "#333"),
        "user_dominant": user_state.get("dominant", ""),
        "model_dominant": model_state.get("dominant", ""),
    }

    return "", history, json.dumps(orb_data), json.dumps(timeline_entry)


# =========================================================================
#  Orb HTML — self-contained Canvas 2D renderer (runs inside gr.HTML)
# =========================================================================

ORB_STRUCTURE = """
<div id="emotionOrbContainer" style="width:100%; min-height:520px; background:#0a0a0f;
     border-radius:12px; padding:16px 16px 8px; display:flex; flex-direction:column;
     align-items:center; gap:6px; font-family:Inter,-apple-system,sans-serif;">

  <!-- User orb -->
  <div style="color:#606078; font-size:10px; text-transform:uppercase; letter-spacing:1px; margin-top:4px;">
    Model's emotional state — processing your message</div>
  <canvas id="userOrbCanvas" width="220" height="180"></canvas>
  <div id="userLabel" style="color:#808098; font-size:12px; min-height:16px;">—</div>

  <!-- Model orb -->
  <div style="color:#606078; font-size:10px; text-transform:uppercase; letter-spacing:1px; margin-top:8px;">
    Model's emotional state — beginning response</div>
  <canvas id="modelOrbCanvas" width="220" height="200"></canvas>
  <div id="modelLabel" style="color:#808098; font-size:12px; min-height:16px;">—</div>

  <!-- Valence strip -->
  <canvas id="valenceStrip" width="200" height="20" style="margin-top:6px;"></canvas>
  <div style="display:flex; justify-content:space-between; width:200px;
       font-size:9px; color:#404058;">
    <span>negative</span><span>neutral</span><span>positive</span>
  </div>

  <!-- Timeline -->
  <canvas id="timelineCanvas" width="200" height="20" style="margin-top:2px;"></canvas>

  <!-- Detail -->
  <div id="orbDetail" style="color:#505068; font-size:10px; font-family:monospace;
       text-align:left; width:200px; min-height:30px; margin-top:2px; line-height:1.5;"></div>
</div>"""

# JavaScript for the orb renderer — injected via js_on_load in Gradio 6+
# (Gradio 6 no longer executes <script> tags inside gr.HTML)
ORB_JS = """
(element) => {
// ── OKLCH → RGB ──
function ok2rgb(L,C,H){
  const h=H*Math.PI/180,a=C*Math.cos(h),b=C*Math.sin(h);
  const l_=L+.3963377774*a+.2158037573*b;
  const m_=L-.1055613458*a-.0638541728*b;
  const s_=L-.0894841775*a-1.291485548*b;
  let r=4.0767416621*l_**3-3.3077115913*m_**3+.2309699292*s_**3;
  let g=-1.2684380046*l_**3+2.6097574011*m_**3-.3413193965*s_**3;
  let bl=-.0041960863*l_**3-.7034186147*m_**3+1.707614701*s_**3;
  const gm=v=>v>.0031308?1.055*v**(1/2.4)-.055:12.92*v;
  return[Math.max(0,Math.min(255,Math.round(gm(r)*255))),
         Math.max(0,Math.min(255,Math.round(gm(g)*255))),
         Math.max(0,Math.min(255,Math.round(gm(bl)*255)))];
}
function rgb(r,g,b,a){return a<1?`rgba(${r},${g},${b},${a.toFixed(3)})`:`rgb(${r},${g},${b})`;}

// ── Noise ──
const P=new Uint8Array(512);
{const p=new Uint8Array(256);for(let i=0;i<256;i++)p[i]=i;
for(let i=255;i>0;i--){const j=Math.floor(Math.random()*(i+1));[p[i],p[j]]=[p[j],p[i]];}
for(let i=0;i<512;i++)P[i]=p[i&255];}
function fade(t){return t*t*t*(t*(t*6-15)+10);}
function lerp(a,b,t){return a+t*(b-a);}
function grad2(h,x,y){const a=h&3;return((a<2?x:-x)+((a&1)?y:-y));}
function noise(x,y){
  const X=Math.floor(x)&255,Y=Math.floor(y)&255,xf=x-Math.floor(x),yf=y-Math.floor(y);
  const u=fade(xf),v=fade(yf);
  return lerp(lerp(grad2(P[P[X]+Y],xf,yf),grad2(P[P[X+1]+Y],xf-1,yf),u),
              lerp(grad2(P[P[X]+Y+1],xf,yf-1),grad2(P[P[X+1]+Y+1],xf-1,yf-1),u),v);
}

// ── Spring ──
class Sp{constructor(v,k=80,d=12){this.t=v;this.v=v;this.vel=0;this.k=k;this.d=d;}
  set(t){this.t=t;}
  step(dt){this.vel+=(-this.k*(this.v-this.t)-this.d*this.vel)*dt;this.v+=this.vel*dt;}}

// ── Particle ──
class Pt{constructor(x,y,r,g,b){this.x=x;this.y=y;this.vx=(Math.random()-.5)*20;
  this.vy=-(15+Math.random()*30);this.r=r;this.g=g;this.b=b;this.life=1;
  this.decay=.6+Math.random()*.5;this.sz=1.5+Math.random()*2;}
  update(dt){this.x+=this.vx*dt;this.y+=this.vy*dt;this.vy-=6*dt;this.life-=this.decay*dt;}
  draw(c){if(this.life<=0)return;const a=this.life*.6,s=this.sz*(.5+this.life*.5);
    const g=c.createRadialGradient(this.x,this.y,0,this.x,this.y,s);
    g.addColorStop(0,rgb(this.r,this.g,this.b,a));g.addColorStop(1,rgb(this.r,this.g,this.b,0));
    c.fillStyle=g;c.beginPath();c.arc(this.x,this.y,s,0,Math.PI*2);c.fill();}}

// ── Single orb renderer ──
class OrbR{
  constructor(canvasId, baseR, root){
    this.cv=(root||document).querySelector('#'+canvasId);
    this.ctx=this.cv.getContext('2d');
    this.W=this.cv.width; this.H=this.cv.height;
    this.cx=this.W/2; this.cy=this.H/2;
    this.baseR=baseR;
    this.NP=60;
    this.sp={hue:new Sp(240,80,12),L:new Sp(.58,100,14),C:new Sp(.03,100,14),
      rad:new Sp(baseR,150,16),flow:new Sp(.3,60,10),dist:new Sp(3,80,12),
      glow:new Sp(.1,80,12),cplx:new Sp(.05,40,8),
      h2:new Sp(240,60,10),L2:new Sp(.5,80,12),C2:new Sp(.05,80,12)};
    this.particles=[];this.pTimer=0;
    this.dominant='—';this.secondary='';
  }

  setState(s){
    if(!s||!s.color_oklch)return;
    const c=s.color_oklch;
    this.sp.hue.set(c.H);this.sp.L.set(c.L);this.sp.C.set(c.C);
    const inten=Math.min(s.intensity||0,1);
    this.sp.rad.set(this.baseR*.5+inten*this.baseR*.7);
    const a01=((s.arousal||0)+1)/2;
    this.sp.flow.set(.12+a01*1.2);this.sp.dist.set(1.5+a01*14);
    this.sp.glow.set(.03+inten*.45);this.sp.cplx.set(s.complexity||0);
    if(s.secondary_oklch){
      this.sp.h2.set(s.secondary_oklch.H);
      this.sp.L2.set(s.secondary_oklch.L);
      this.sp.C2.set(s.secondary_oklch.C);}
    this.dominant=s.dominant||'—';
    this.secondary=s.secondary||'';
  }

  step(dt,t){
    for(const s of Object.values(this.sp))s.step(dt);
    // particles
    const a01=(this.sp.flow.v-.12)/1.2;
    const rate=a01>.55?(a01-.55)/.45*14:0;
    this.pTimer+=dt;const iv=rate>0?1/rate:999;
    while(this.pTimer>iv&&rate>0){this.pTimer-=iv;
      const ang=Math.random()*Math.PI*2,r=this.sp.rad.v*(.8+Math.random()*.25);
      const[cr,cg,cb]=ok2rgb(this.sp.L.v,this.sp.C.v,this.sp.hue.v);
      this.particles.push(new Pt(this.cx+Math.cos(ang)*r,this.cy+Math.sin(ang)*r,cr,cg,cb));}
    for(const p of this.particles)p.update(dt);
    this.particles=this.particles.filter(p=>p.life>0);
  }

  draw(t){
    const ctx=this.ctx,R=this.sp.rad.v,fl=this.sp.flow.v,di=this.sp.dist.v;
    const cp=this.sp.cplx.v,gl=this.sp.glow.v;
    const[pr,pg,pb]=ok2rgb(this.sp.L.v,this.sp.C.v,this.sp.hue.v);
    ctx.clearRect(0,0,this.W,this.H);

    // glow
    const gR=R*(1.4+gl*1.5);
    const gg=ctx.createRadialGradient(this.cx,this.cy,R*.2,this.cx,this.cy,gR);
    gg.addColorStop(0,rgb(pr,pg,pb,gl*.3));gg.addColorStop(.5,rgb(pr,pg,pb,gl*.1));
    gg.addColorStop(1,rgb(pr,pg,pb,0));ctx.fillStyle=gg;
    ctx.fillRect(0,0,this.W,this.H);

    // blob
    const pts=[];
    for(let i=0;i<this.NP;i++){
      const th=i/this.NP*Math.PI*2;
      const n1=noise(th*1.5+t*fl,t*fl*.3+10)*di;
      const n2=noise(th*2.3-t*fl*.7,t*fl*.5+50)*di*.6;
      const n3=noise(th*3.7+t*fl*1.5,t*fl*.9+90)*di*.35*cp;
      pts.push({x:this.cx+Math.cos(th)*(R+n1+n2+n3),
                y:this.cy+Math.sin(th)*(R+n1+n2+n3)});}

    ctx.save();ctx.beginPath();
    for(let i=0;i<this.NP;i++){
      const p0=pts[(i-1+this.NP)%this.NP],p1=pts[i],
            p2=pts[(i+1)%this.NP],p3=pts[(i+2)%this.NP];
      if(i===0)ctx.moveTo(p1.x,p1.y);
      ctx.bezierCurveTo(p1.x+(p2.x-p0.x)/6,p1.y+(p2.y-p0.y)/6,
                        p2.x-(p3.x-p1.x)/6,p2.y-(p3.y-p1.y)/6,p2.x,p2.y);}
    ctx.closePath();

    // body gradient
    const cL=Math.min(this.sp.L.v+.1,.92);
    const[cr,cg,cb]=ok2rgb(cL,this.sp.C.v*1.1,this.sp.hue.v);
    const bg=ctx.createRadialGradient(this.cx-R*.12,this.cy-R*.12,R*.04,this.cx,this.cy,R*1.05);
    bg.addColorStop(0,rgb(cr,cg,cb,.92));
    if(cp>.3){const bl=Math.min((cp-.3)/.5,.55);
      const[sr,sg,sb]=ok2rgb(this.sp.L2.v,this.sp.C2.v,this.sp.h2.v);
      bg.addColorStop(.4,rgb(Math.round(pr*(1-bl)+sr*bl),Math.round(pg*(1-bl)+sg*bl),
                            Math.round(pb*(1-bl)+sb*bl),.88));}
    bg.addColorStop(.82,rgb(pr,pg,pb,.82));bg.addColorStop(1,rgb(pr,pg,pb,.25));
    ctx.fillStyle=bg;ctx.fill();

    // specular
    const sx=this.cx-R*.25,sy=this.cy-R*.25;
    const sg2=ctx.createRadialGradient(sx,sy,0,sx,sy,R*.4);
    sg2.addColorStop(0,`rgba(255,255,255,${.09+gl*.12})`);
    sg2.addColorStop(.6,`rgba(255,255,255,${.02+gl*.03})`);
    sg2.addColorStop(1,'rgba(255,255,255,0)');ctx.fillStyle=sg2;ctx.fill();

    // secondary current
    if(cp>.3){const[sr,sg,sb]=ok2rgb(this.sp.L2.v,this.sp.C2.v,this.sp.h2.v);
      const bl=Math.min((cp-.3)/.5,.45);
      const ox=noise(t*fl*.4,200)*R*.4,oy=noise(t*fl*.4+100,200)*R*.4;
      const sc=ctx.createRadialGradient(this.cx+ox,this.cy+oy,0,this.cx+ox,this.cy+oy,R*.6);
      sc.addColorStop(0,rgb(sr,sg,sb,bl*.4));sc.addColorStop(.6,rgb(sr,sg,sb,bl*.15));
      sc.addColorStop(1,rgb(sr,sg,sb,0));ctx.fillStyle=sc;ctx.fill();}
    ctx.restore();

    // particles
    for(const p of this.particles)p.draw(ctx);
  }
}

// ── Create the two orbs ──
// In Gradio 6 js_on_load, `element` is the HTML component's DOM node
const userOrb=new OrbR('userOrbCanvas',45,element);
const modelOrb=new OrbR('modelOrbCanvas',55,element);

// ── Valence strip ──
function drawValenceStrip(val){
  const vc=element.querySelector('#valenceStrip');
  if(!vc)return;
  const vctx=vc.getContext('2d');
  const w=vc.width,h=vc.height;
  vctx.clearRect(0,0,w,h);
  const stops=[[0,.25,.14,15],[.15,.30,.18,10],[.3,.40,.10,270],
    [.5,.58,.03,240],[.7,.65,.08,220],[.85,.75,.10,85],[1,.82,.15,85]];
  const gr=vctx.createLinearGradient(0,0,w,0);
  for(const[pos,L,C,H] of stops){const[r,g,b]=ok2rgb(L,C,H);gr.addColorStop(pos,rgb(r,g,b,1));}
  vctx.fillStyle=gr;
  vctx.beginPath();vctx.roundRect(0,0,w,h,3);vctx.fill();
  // marker
  const mx=Math.max(2,Math.min(w-2,(val+1)/2*w));
  vctx.fillStyle='#fff';vctx.shadowColor='#fff';vctx.shadowBlur=5;
  vctx.beginPath();vctx.roundRect(mx-1.5,1,3,h-2,1.5);vctx.fill();
  vctx.shadowBlur=0;
}

// ── Timeline ──
let timeline=[];
function drawTimeline(){
  const tc=element.querySelector('#timelineCanvas');
  if(!tc)return;
  const tctx=tc.getContext('2d'),w=tc.width,h=tc.height;
  tctx.fillStyle='#12121a';tctx.beginPath();tctx.roundRect(0,0,w,h,3);tctx.fill();
  if(!timeline.length)return;
  const segW=Math.max(6,Math.min(w/timeline.length,30));
  const startX=w-timeline.length*segW;
  for(let i=0;i<timeline.length;i++){
    const t=timeline[i];
    // top half = user, bottom half = model
    tctx.fillStyle=t.user_color||'#333';
    tctx.fillRect(startX+i*segW+.5,1,segW-1,h/2-1);
    tctx.fillStyle=t.model_color||'#333';
    tctx.fillRect(startX+i*segW+.5,h/2,segW-1,h/2-1);
  }
}

// ── Animation loop ──
let time=0,lastT=performance.now();
let curModelState={},curUserState={};

function loop(now){
  const dt=Math.min((now-lastT)/1000,.05);lastT=now;time+=dt;
  userOrb.step(dt,time);modelOrb.step(dt,time);
  userOrb.draw(time);modelOrb.draw(time);
  drawValenceStrip(curModelState.valence||0);
  drawTimeline();

  // labels
  const ul=element.querySelector('#userLabel');
  const ml=element.querySelector('#modelLabel');
  if(ul){let t=curUserState.dominant||'—';
    if(curUserState.secondary&&(curUserState.complexity||0)>.25)t+=' · '+curUserState.secondary;
    ul.textContent=t;}
  if(ml){let t=curModelState.dominant||'—';
    if(curModelState.secondary&&(curModelState.complexity||0)>.25)t+=' · '+curModelState.secondary;
    ml.textContent=t;}

  // detail
  const det=element.querySelector('#orbDetail');
  if(det&&curModelState.top_emotions){
    det.innerHTML=curModelState.top_emotions.slice(0,4)
      .map(([n,s])=>`<span style="color:#8888aa">${n}</span> ${s.toFixed(2)}`).join('&ensp;')
      +'<br><span style="color:#505068">v='+(curModelState.valence||0).toFixed(2)
      +' a='+(curModelState.arousal||0).toFixed(2)+'</span>';}

  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// ── Receive state updates from Gradio ──
window._updateOrb=function(jsonStr){
  try{
    const d=JSON.parse(jsonStr);
    if(d.model){curModelState=d.model;modelOrb.setState(d.model);}
    if(d.user){curUserState=d.user;userOrb.setState(d.user);}
    console.log('[orb] state received',d);
  }catch(e){console.error('[orb] parse error',e);}
};
window._addTimeline=function(jsonStr){
  try{timeline.push(JSON.parse(jsonStr));if(timeline.length>30)timeline.shift();}catch(e){}
};
}
"""


# =========================================================================
#  Gradio app
# =========================================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(title="EmotionScope") as app:
        gr.Markdown("# EmotionScope\n*Real-time emotion indicators from the model's residual stream*")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500, label="Conversation")
                msg = gr.Textbox(
                    placeholder="Type a message...",
                    show_label=False, container=False)

            with gr.Column(scale=1, min_width=280):
                orb_html = gr.HTML(value=ORB_STRUCTURE, js_on_load=ORB_JS)

        # Hidden state holders — JS polls these for updates
        orb_state = gr.Textbox(visible=False, elem_id="orbState")
        timeline_state = gr.Textbox(visible=False, elem_id="timelineState")

        msg.submit(
            fn=chat_fn,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, orb_state, timeline_state],
        )

        # JS polling loop: watches the hidden textboxes for changes and
        # pushes data into the orb renderer. More reliable than Gradio's
        # .then(js=...) chain which doesn't fire in Gradio 6.x.
        app.load(js="""
        () => {
            let lastOrb = '', lastTl = '';
            function poll() {
                try {
                    const orbEl = document.querySelector('#orbState textarea');
                    const tlEl = document.querySelector('#timelineState textarea');
                    if (orbEl && orbEl.value && orbEl.value !== lastOrb) {
                        lastOrb = orbEl.value;
                        if (window._updateOrb) window._updateOrb(lastOrb);
                    }
                    if (tlEl && tlEl.value && tlEl.value !== lastTl) {
                        lastTl = tlEl.value;
                        if (window._addTimeline) window._addTimeline(lastTl);
                    }
                } catch(e) {}
                requestAnimationFrame(poll);
            }
            setTimeout(poll, 1500);
        }
        """)

    return app


def main():
    parser = argparse.ArgumentParser(description="EmotionScope Gradio Demo")
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--vectors", default="results/vectors/google_gemma-2-2b-it.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    vectors_path = Path(args.vectors)
    if not vectors_path.exists():
        # Auto-download from HuggingFace Hub
        print(f"[demo] Vectors not found at {vectors_path}, trying Hub download...")
        try:
            from emotion_scope.hub import download_vectors
            vectors_path = download_vectors(args.model)
        except Exception:
            print(f"[demo] Download failed. Extract locally with:")
            print(f"  uv run python scripts/extract_all.py --model {args.model} --sweep-layers")
            return

    init_probe(args.model, str(vectors_path), args.device)
    app = build_app()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
