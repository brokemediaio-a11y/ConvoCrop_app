'use client'

import React, { useEffect, useRef, useState } from 'react'
import {
  Scene,
  PerspectiveCamera,
  WebGLRenderer,
  Color,
  AmbientLight,
  DirectionalLight,
  PointLight,
  Clock,
  Group,
  Vector3,
  Box3,
  Object3D,
} from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import gsap from 'gsap'

const MODEL_PATH = '/rice_plant/Untitled.glb'

/** Fixed width for hero message stack (matches original layout) */
const PANEL_WIDTH = 260

const PHASES = [
  {
    rotationY: -0.5,
    userText: 'What disease do you see in this plant?',
    aiText:
      'I can identify signs of rice blast caused by Magnaporthe oryzae. The spindle-shaped lesions with pale centers and darker borders are typical on infected leaves.',
  },
  {
    rotationY: 0.6,
    userText: 'What are the causes of this disease?',
    aiText:
      'Rice blast spreads through airborne spores and favors warm, humid weather with heavy dew. Dense planting and excess nitrogen can increase risk.',
  },
]

export default function PlantViewer() {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLDivElement>(null)
  const pathRef = useRef<SVGPathElement>(null)
  const dotRef = useRef<SVGCircleElement>(null)
  const pulseRef = useRef<SVGCircleElement>(null)
  const userBoxRef = useRef<HTMLDivElement>(null)
  const userTextRef = useRef<HTMLParagraphElement>(null)
  const aiBoxRef = useRef<HTMLDivElement>(null)
  const aiTextRef = useRef<HTMLParagraphElement>(null)
  const textContainerRef = useRef<HTMLDivElement>(null)
  const currentLeafRef = useRef(0)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    const canvasEl = canvasRef.current
    const containerEl = containerRef.current
    if (!canvasEl || !containerEl) return

    const scene = new Scene()
    scene.background = null

    const camera = new PerspectiveCamera(38, 1, 0.01, 200)

    const renderer = new WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setClearColor(new Color(0x000000), 0)
    renderer.outputColorSpace = 'srgb'
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    canvasEl.appendChild(renderer.domElement)

    scene.add(new AmbientLight(0xffffff, 0.9))
    const keyLight = new DirectionalLight(0xffffff, 1.5)
    keyLight.position.set(3, 5, 4)
    scene.add(keyLight)
    const fillLight = new DirectionalLight(0x88ffaa, 0.5)
    fillLight.position.set(-3, 2, -2)
    scene.add(fillLight)
    const rimLight = new PointLight(0xa8ffbf, 0.6, 10)
    rimLight.position.set(-2, 3, -3)
    scene.add(rimLight)

    const clock = new Clock()
    let plantRoot: Group | null = null
    let leafObjects: Object3D[] = []
    let entranceDone = false
    let finalScale = 1
    let finalY = 0
    let baseCamZ = 5
    let camLookY = 1
    let basePlantX = 0
    let plantHeight = 0
    let tl: gsap.core.Timeline | null = null
    const typingIntervals: number[] = []

    let leavesNode: Object3D | null = null
    let baseRZ = 0
    let baseRX = 0
    let curRZ = 0
    let curRX = 0

    function clearTyping() {
      typingIntervals.forEach((id) => window.clearInterval(id))
      typingIntervals.length = 0
    }

    function typeText(el: HTMLElement | null, text: string, speed: number) {
      if (!el) return
      el.textContent = ''
      let i = 0
      const id = window.setInterval(() => {
        i++
        el.textContent = text.slice(0, i)
        if (i >= text.length) {
          window.clearInterval(id)
          const idx = typingIntervals.indexOf(id)
          if (idx !== -1) typingIntervals.splice(idx, 1)
        }
      }, speed)
      typingIntervals.push(id)
    }

    function projectToScreen(worldPos: Vector3) {
      const v = worldPos.clone().project(camera)
      const w = canvasEl?.clientWidth || 1
      const h = canvasEl?.clientHeight || 1
      return {
        x: (v.x * 0.5 + 0.5) * w,
        y: (-v.y * 0.5 + 0.5) * h,
      }
    }

    function easeOutCubic(t: number) {
      return 1 - Math.pow(1 - t, 3)
    }

    function findLeafAnchors() {
      if (!plantRoot) return

      const candidates: { obj: Object3D; pos: Vector3 }[] = []

      plantRoot.traverse((child: any) => {
        if (!child.isMesh || !child.parent) return
        const pName = child.parent.name || ''
        if (!pName.startsWith('Plane')) return

        const wp = new Vector3()
        child.getWorldPosition(wp)
        candidates.push({ obj: child, pos: wp })
      })

      if (candidates.length < 2) {
        plantRoot.traverse((child: any) => {
          if (child.isMesh) {
            const wp = new Vector3()
            child.getWorldPosition(wp)
            candidates.push({ obj: child, pos: wp })
          }
        })
      }

      if (candidates.length < 2) return

      candidates.sort((a, b) => b.pos.y - a.pos.y)
      const dropStem = Math.max(1, Math.floor(candidates.length * 0.28))
      const foliage = candidates.slice(0, candidates.length - dropStem)

      const pool = foliage.length >= 2 ? foliage : candidates
      pool.sort((a, b) => a.pos.x - b.pos.x)
      leafObjects = [pool[0].obj, pool[pool.length - 1].obj]
    }

    function leafBladeScreenPoint(obj: Object3D): { x: number; y: number } {
      const box = new Box3().setFromObject(obj)
      const wp = new Vector3()
      if (box.isEmpty()) {
        obj.getWorldPosition(wp)
      } else {
        const size = new Vector3()
        box.getCenter(wp)
        box.getSize(size)
        wp.y += size.y * 0.22
      }
      return projectToScreen(wp)
    }

    function computeDynamicX() {
      if (!plantRoot) return
      const vFov = camera.fov * Math.PI / 180
      const visHeight = 2 * Math.tan(vFov / 2) * camera.position.z
      const visWidth = visHeight * camera.aspect
      plantRoot.position.x = basePlantX + visWidth * 0.2
    }

    const START_ROTATION = 0

    function buildTimeline() {
      if (!plantRoot || leafObjects.length < 2) return

      plantRoot.rotation.y = START_ROTATION
      camera.position.z = baseCamZ
      camera.lookAt(0, camLookY, 0)

      const zoomedZ = baseCamZ * 0.8

      tl = gsap.timeline({ repeat: -1 })

      PHASES.forEach((phase, phaseIdx) => {
        const leafIdx = phaseIdx % leafObjects.length
        const rotLabel = `phase${phaseIdx}`

        tl!.call(() => {
          currentLeafRef.current = leafIdx
        })

        tl!.to(plantRoot!.rotation, {
          y: phase.rotationY,
          duration: 1.6,
          ease: 'power2.inOut',
        }, rotLabel)

        tl!.to(camera.position, {
          z: zoomedZ,
          duration: 1.6,
          ease: 'power2.inOut',
          onUpdate: () => { camera.lookAt(0, camLookY, 0) },
        }, rotLabel)

        tl!.call(
          () => {
            if (dotRef.current) gsap.to(dotRef.current, { opacity: 1, duration: 0.4 })
            if (pulseRef.current) gsap.to(pulseRef.current, { opacity: 1, duration: 0.4 })
            if (pathRef.current) {
              const len = pathRef.current.getTotalLength() || 200
              gsap.set(pathRef.current, {
                strokeDasharray: len,
                strokeDashoffset: len,
                opacity: 0.9,
              })
              gsap.to(pathRef.current, {
                strokeDashoffset: 0,
                duration: 0.7,
                ease: 'power1.out',
              })
            }
          },
          undefined,
          '+=0.1',
        )

        tl!.call(
          () => {
            if (userBoxRef.current)
              gsap.to(userBoxRef.current, { opacity: 1, y: 0, duration: 0.35, ease: 'power2.out' })
            typeText(userTextRef.current, phase.userText, 28)
          },
          undefined,
          '+=0.5',
        )

        tl!.call(
          () => {
            if (aiBoxRef.current)
              gsap.to(aiBoxRef.current, { opacity: 1, y: 0, duration: 0.35, ease: 'power2.out' })
            typeText(aiTextRef.current, phase.aiText, 18)
          },
          undefined,
          '+=1.4',
        )

        tl!.to({}, { duration: 3.2 })

        tl!.call(() => {
          clearTyping()
          const els = [
            userBoxRef.current,
            aiBoxRef.current,
            pathRef.current,
            dotRef.current,
            pulseRef.current,
          ]
          els.forEach((el) => {
            if (el) gsap.to(el, { opacity: 0, duration: 0.5 })
          })
        })

        tl!.to(camera.position, {
          z: baseCamZ,
          duration: 1.0,
          ease: 'power2.inOut',
          onUpdate: () => { camera.lookAt(0, camLookY, 0) },
        }, '+=0.3')

        tl!.call(() => {
          if (userBoxRef.current) gsap.set(userBoxRef.current, { y: 10, opacity: 0 })
          if (aiBoxRef.current) gsap.set(aiBoxRef.current, { y: 10, opacity: 0 })
          if (userTextRef.current) userTextRef.current.textContent = ''
          if (aiTextRef.current) aiTextRef.current.textContent = ''
        })
      })

      tl!.to(plantRoot!.rotation, {
        y: START_ROTATION,
        duration: 1.4,
        ease: 'power2.inOut',
      })
    }

    const loader = new GLTFLoader()
    loader.load(MODEL_PATH, (gltf) => {
      plantRoot = gltf.scene as Group

      plantRoot.traverse((child: any) => {
        if (child.name === 'Leaves' && !leavesNode) leavesNode = child
      })

      if (!leavesNode) {
        plantRoot.traverse((child: any) => {
          const n = (child.name || '').toLowerCase()
          if ((n.includes('leaves') || n.includes('leaf')) && !leavesNode) leavesNode = child
        })
      }

      if (leavesNode) {
        baseRZ = leavesNode.rotation.z
        baseRX = leavesNode.rotation.x
        curRZ = baseRZ
        curRX = baseRX
      }

      const box = new Box3().setFromObject(plantRoot)
      const size = new Vector3()
      const center = new Vector3()
      box.getSize(size)
      box.getCenter(center)

      const maxDim = Math.max(size.x, size.y, size.z)
      const desiredSize = 5.5
      finalScale = desiredSize / maxDim

      plantHeight = size.y * finalScale
      const baseY = -box.min.y * finalScale
      const yOffset = -0.8
      finalY = baseY + yOffset
      basePlantX = -center.x * finalScale

      camLookY = baseY + plantHeight * 0.45
      const fovRad = camera.fov * Math.PI / 360
      const aboveCenter = (finalY + plantHeight) - camLookY
      const belowCenter = camLookY - finalY
      const maxExtent = Math.max(aboveCenter, belowCenter) * 1.1
      baseCamZ = maxExtent / Math.tan(fovRad)

      plantRoot.scale.setScalar(0.001)
      plantRoot.position.set(
        basePlantX,
        finalY - 1.5,
        -center.z * finalScale,
      )

      camera.position.set(0, camLookY + 0.2, baseCamZ)
      camera.lookAt(0, camLookY, 0)

      scene.add(plantRoot)
      clock.start()
      setLoaded(true)
    })

    const resize = () => {
      const w = canvasEl.clientWidth || 1
      const h = canvasEl.clientHeight || 1
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h, false)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvasEl)

    let frameId: number
    const animate = () => {
      const elapsed = clock.getElapsedTime()

      if (plantRoot) {
        computeDynamicX()

        if (!entranceDone) {
          const p = Math.min(elapsed / 1.8, 1)
          const e = easeOutCubic(p)
          plantRoot.scale.setScalar(0.001 + (finalScale - 0.001) * e)
          plantRoot.position.y = finalY - 1.5 + 1.5 * e

          if (p >= 1) {
            entranceDone = true
            plantRoot.scale.setScalar(finalScale)
            plantRoot.position.y = finalY
            findLeafAnchors()
            buildTimeline()
          }
        }

        if (leavesNode && entranceDone) {
          const amp = 0.012
          const spd = 0.4
          const tZ = baseRZ + Math.sin(elapsed * spd) * amp
          const tX = baseRX + Math.cos(elapsed * spd * 0.7) * amp * 0.5
          curRZ += (tZ - curRZ) * 0.05
          curRX += (tX - curRX) * 0.05
          leavesNode.rotation.z = curRZ
          leavesNode.rotation.x = curRX
        }

        if (entranceDone && leafObjects.length > 0) {
          const leafObj = leafObjects[currentLeafRef.current]
          if (leafObj) {
            const sp = leafBladeScreenPoint(leafObj)

            if (dotRef.current) {
              dotRef.current.setAttribute('cx', String(sp.x))
              dotRef.current.setAttribute('cy', String(sp.y))
            }
            if (pulseRef.current) {
              pulseRef.current.setAttribute('cx', String(sp.x))
              pulseRef.current.setAttribute('cy', String(sp.y))
            }

            if (textContainerRef.current) {
              const cw = containerEl.clientWidth
              const ch = containerEl.clientHeight
              // Place boxes starting at 55% of container width so they sit
              // clearly in the right half with enough space for full text
              const left = Math.max(10, cw * 0.8)
              const top = ch * 0.22
              textContainerRef.current.style.width = `${Math.min(PANEL_WIDTH, cw - left - 8)}px`
              textContainerRef.current.style.left = `${left}px`
              textContainerRef.current.style.top = `${top}px`
            }

            let textX = containerEl.clientWidth * 0.55
            let textY = containerEl.clientHeight * 0.25

            if (userBoxRef.current) {
              const boxRect = userBoxRef.current.getBoundingClientRect()
              const contRect = containerEl.getBoundingClientRect()
              if (boxRect.width > 0) {
                // Attach line to the left edge, vertically centred on the user box
                textX = boxRect.left - contRect.left
                textY = boxRect.top - contRect.top + boxRect.height * 0.5
              }
            }

            if (pathRef.current) {
              // Control point: pull horizontally toward the box then up slightly
              const mx = sp.x + (textX - sp.x) * 0.45
              const my = Math.min(sp.y, textY) - 20
              pathRef.current.setAttribute(
                'd',
                `M ${sp.x},${sp.y} Q ${mx},${my} ${textX},${textY}`,
              )
            }
          }
        }
      }

      renderer.render(scene, camera)
      frameId = requestAnimationFrame(animate)
    }
    frameId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(frameId)
      if (tl) tl.kill()
      clearTyping()
      gsap.killTweensOf([
        dotRef.current,
        pulseRef.current,
        pathRef.current,
        userBoxRef.current,
        aiBoxRef.current,
      ])
      ro.disconnect()
      if (plantRoot) scene.remove(plantRoot)
      renderer.dispose()
      if (renderer.domElement.parentElement) {
        renderer.domElement.parentElement.removeChild(renderer.domElement)
      }
    }
  }, [])

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <div
        ref={canvasRef}
        className="absolute inset-0 transition-opacity duration-700"
        style={{ opacity: loaded ? 1 : 0 }}
      />

      <div
        className="absolute inset-0 pointer-events-none overflow-visible"
        style={{ zIndex: 2, opacity: loaded ? 1 : 0, transition: 'opacity 0.7s' }}
      >
        <svg className="absolute inset-0 w-full h-full overflow-visible">
          <defs>
            <filter id="lineGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <path
            ref={pathRef}
            fill="none"
            stroke="#00a71b"
            strokeWidth="1.5"
            strokeLinecap="round"
            filter="url(#lineGlow)"
            style={{ opacity: 0 }}
          />

          <circle ref={dotRef} r="3.5" fill="#00a71b" style={{ opacity: 0 }} />

          <circle
            ref={pulseRef}
            r="3.5"
            fill="none"
            stroke="#00a71b"
            strokeWidth="1"
            style={{ opacity: 0 }}
          >
            <animate attributeName="r" values="3.5;14;3.5" dur="2s" repeatCount="indefinite" />
            <animate
              attributeName="stroke-opacity"
              values="0.7;0;0.7"
              dur="2s"
              repeatCount="indefinite"
            />
          </circle>
        </svg>

        <div ref={textContainerRef} className="absolute" style={{ width: PANEL_WIDTH }}>
          <div ref={userBoxRef} className="mb-3" style={{ opacity: 0, transform: 'translateY(10px)' }}>
            <div className="relative px-4 py-3" style={{ background: 'rgba(255,255,255,0.05)' }}>
              <div className="absolute inset-0 border border-[#252525]/70 pointer-events-none" />
              <div className="absolute top-0 left-0 w-2.5 h-2.5 border-t-2 border-l-2 border-[#00a71b]" />
              <div className="absolute top-0 right-0 w-2.5 h-2.5 border-t-2 border-r-2 border-[#00a71b]" />
              <div className="absolute bottom-0 left-0 w-2.5 h-2.5 border-b-2 border-l-2 border-[#00a71b]" />
              <div className="absolute bottom-0 right-0 w-2.5 h-2.5 border-b-2 border-r-2 border-[#00a71b]" />
              <span className="text-[#00a71b] text-[11px] font-bold tracking-[0.15em] uppercase block mb-1.5">
                You
              </span>
              <p ref={userTextRef} className="text-[#111111] text-xs font-medium leading-relaxed m-0" />
            </div>
          </div>

          <div ref={aiBoxRef} style={{ opacity: 0, transform: 'translateY(10px)' }}>
            <div className="relative px-4 py-3" style={{ background: 'rgba(255,255,255,0.05)' }}>
              <div className="absolute inset-0 border border-[#252525]/70 pointer-events-none" />
              <div className="absolute top-0 left-0 w-2.5 h-2.5 border-t-2 border-l-2 border-[#00a71b]" />
              <div className="absolute top-0 right-0 w-2.5 h-2.5 border-t-2 border-r-2 border-[#00a71b]" />
              <div className="absolute bottom-0 left-0 w-2.5 h-2.5 border-b-2 border-l-2 border-[#00a71b]" />
              <div className="absolute bottom-0 right-0 w-2.5 h-2.5 border-b-2 border-r-2 border-[#00a71b]" />
              <span className="text-[#00a71b] text-[11px] font-bold tracking-[0.15em] uppercase block mb-1.5">
                Convo Crop
              </span>
              <p
                ref={aiTextRef}
                className="text-xs font-medium leading-relaxed m-0 text-[#111111]"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}