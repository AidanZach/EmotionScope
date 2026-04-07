import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { MeshDistortMaterial, Float, Sparkles, Environment } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import { useSpring, animated } from '@react-spring/three'
import * as THREE from 'three'
import { emotionToOrbProps } from '../utils/palette'

const AnimatedMeshDistortMaterial = animated(MeshDistortMaterial)

// ── Inner Core ──────────────────────────────────────────────
// A small bright sphere at the center, visible through the translucent
// outer shell. Creates the illusion of depth — light comes from INSIDE.
// The core is the "truth" of the emotion; the shell is the surface.
function InnerCore({ color, intensity }) {
  const ref = useRef()

  useFrame((state) => {
    if (ref.current) {
      // Gentle pulsing glow — like a heartbeat
      const pulse = Math.sin(state.clock.elapsedTime * 1.5) * 0.08
      ref.current.material.emissiveIntensity = 0.6 + intensity * 0.8 + pulse
    }
  })

  return (
    <mesh ref={ref} scale={0.28}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.8}
        transparent
        opacity={0.7 + intensity * 0.25}
        toneMapped={false}
      />
    </mesh>
  )
}

// ── Atmosphere ──────────────────────────────────────────────
// A larger, very transparent sphere that connects the orb to its space.
// The atmosphere is what makes particles feel "part of" the orb —
// they exist WITHIN the atmosphere, not randomly near a solid ball.
function Atmosphere({ color, intensity, arousal }) {
  const ref = useRef()

  useFrame((state) => {
    if (ref.current) {
      // Slow breathing — the atmosphere expands and contracts gently
      const breathe = Math.sin(state.clock.elapsedTime * 0.8) * 0.03
      ref.current.scale.setScalar(1.35 + breathe + intensity * 0.1)
      ref.current.material.opacity = 0.03 + intensity * 0.05
    }
  })

  return (
    <mesh ref={ref} scale={1.35}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.04}
        side={THREE.BackSide}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  )
}

// ── Main Orb Mesh ───────────────────────────────────────────
function OrbMesh({ emotionState }) {
  const meshRef = useRef()
  const props = emotionToOrbProps(emotionState)

  const springs = useSpring({
    scale: props.scale,
    distort: props.distort,
    speed: props.speed,
    emissiveIntensity: 0.3 + props.intensity * 0.9,
    opacity: 0.82 + props.intensity * 0.1,
    color: props.color,
    emissive: props.emissive,
    config: (key) => {
      switch (key) {
        case 'scale':             return { mass: 1, tension: 170, friction: 16 }
        case 'color':             return { mass: 1, tension: 80, friction: 12 }
        case 'emissive':          return { mass: 1, tension: 80, friction: 12 }
        case 'distort':           return { mass: 2, tension: 40, friction: 8 }
        case 'speed':             return { mass: 1.5, tension: 60, friction: 10 }
        case 'emissiveIntensity': return { mass: 1, tension: 120, friction: 14 }
        default:                  return { mass: 1, tension: 120, friction: 14 }
      }
    },
  })

  useFrame((state, delta) => {
    if (meshRef.current) {
      // Autonomous rotation — reflections swim across the distorted surface
      meshRef.current.rotation.y += delta * 0.15
      meshRef.current.rotation.x += delta * 0.07

      // Breathing pulse on emissive — subtle, tied to a slow rhythm
      // Makes the orb feel alive even when nothing is changing
      const breathe = Math.sin(state.clock.elapsedTime * 1.2) * 0.06
      if (meshRef.current.material?.emissiveIntensity !== undefined) {
        meshRef.current.material.emissiveIntensity += breathe * delta * 2
      }
    }
  })

  return (
    <animated.mesh ref={meshRef} scale={springs.scale}>
      <sphereGeometry args={[1, 128, 128]} />
      <AnimatedMeshDistortMaterial
        color={springs.color}
        emissive={springs.emissive}
        emissiveIntensity={springs.emissiveIntensity}
        distort={springs.distort}
        speed={springs.speed}
        roughness={0.03}
        metalness={0.6}
        clearcoat={1.0}
        clearcoatRoughness={0.05}
        envMapIntensity={1.8}
        sheen={0.4}
        sheenRoughness={0.2}
        sheenColor={springs.emissive}
        transparent
        opacity={springs.opacity}
      />
    </animated.mesh>
  )
}

// ── Full Orb Scene ──────────────────────────────────────────
export default function EmotionOrbCanvas({ emotionState, size = 200 }) {
  const arousal = emotionState?.arousal ?? 0
  const intensity = emotionState?.intensity ?? 0.3
  const props = emotionToOrbProps(emotionState)

  // CSS glow — the orb's light bleeds into the dark panel around it.
  // A very subtle box-shadow in the emotion's color. This connects the
  // 3D canvas to the 2D UI — the orb isn't trapped in a box, it stains
  // its surroundings.
  const glowStyle = useMemo(() => {
    const glowRadius = 20 + intensity * 30
    const glowOpacity = 0.08 + intensity * 0.15
    return {
      width: size,
      height: size,
      filter: `drop-shadow(0 0 ${glowRadius}px color-mix(in srgb, ${props.color} ${Math.round(glowOpacity * 100)}%, transparent))`,
    }
  }, [size, props.color, intensity])

  return (
    <div className="orb-canvas-wrapper" style={glowStyle}>
      <Canvas
        camera={{ position: [0, 0, 3.5], fov: 45 }}
        gl={{ alpha: true, antialias: true, toneMapping: 3 }}
        style={{ background: 'transparent' }}
      >
        <Environment preset="night" backgroundIntensity={0} />

        <ambientLight intensity={0.08} />
        <pointLight position={[3, 3, 5]} intensity={0.8} color="#fff5ee" />
        <pointLight position={[-3, -1, 3]} intensity={0.5} color={props.color} />
        <pointLight position={[0, -3, -2]} intensity={0.35} color={props.color} />

        <Float
          speed={1 + arousal * 0.5}
          rotationIntensity={0.15}
          floatIntensity={0.2 + Math.max(0, arousal) * 0.15}
        >
          {/* The layered orb: core → shell → atmosphere */}
          <InnerCore color={props.color} intensity={intensity} />
          <OrbMesh emotionState={emotionState} />
          <Atmosphere color={props.color} intensity={intensity} arousal={arousal} />

          {/* Ambient motes — always visible, very faint.
              These give the scene depth and make the orb feel like it
              exists in a space, not on a stage. Like dust in a sunbeam. */}
          <Sparkles
            count={8}
            scale={2.2}
            size={0.8}
            speed={0.15}
            color={props.color}
            opacity={0.15}
          />

          {/* High-arousal particles — tighter to the surface now.
              Scale 1.6 (was 2.5+) keeps them close to the orb radius
              so they feel shed FROM the surface, not floating nearby.
              Slower, fewer, more deliberate — each one visible. */}
          {arousal > 0.4 && (
            <Sparkles
              count={Math.floor(10 + arousal * 25)}
              scale={1.6 + intensity * 0.4}
              size={1.2 + arousal * 1.5}
              speed={0.2 + arousal * 0.3}
              color={props.color}
              opacity={0.35 + arousal * 0.35}
            />
          )}
        </Float>

        <EffectComposer>
          <Bloom
            luminanceThreshold={0.1}
            luminanceSmoothing={0.8}
            intensity={0.6 + intensity * 1.2}
            radius={0.85}
          />
        </EffectComposer>
      </Canvas>
    </div>
  )
}
