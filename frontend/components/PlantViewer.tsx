'use client'

import React, { useEffect, useRef } from 'react'
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
  Vector2,
  Vector3,
  Box3,
  Raycaster,
  Mesh,
  Object3D,
} from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

const MODEL_PATH = '/3d_model/scene.gltf'

export default function PlantViewer() {
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const scene = new Scene()
    scene.background = null

    const camera = new PerspectiveCamera(40, 1, 0.01, 200)
    camera.position.set(0, 1.5, 4)
    camera.lookAt(0, 0.8, 0)

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
    container.appendChild(renderer.domElement)

    // Lighting
    const ambient = new AmbientLight(0xffffff, 0.6)
    scene.add(ambient)

    const keyLight = new DirectionalLight(0xffffff, 1.2)
    keyLight.position.set(3, 5, 4)
    scene.add(keyLight)

    const fillLight = new DirectionalLight(0x88ffaa, 0.4)
    fillLight.position.set(-3, 2, -2)
    scene.add(fillLight)

    const rimLight = new PointLight(0xa8ffbf, 0.6, 10)
    rimLight.position.set(-2, 3, -3)
    scene.add(rimLight)

    const clock = new Clock()
    const raycaster = new Raycaster()
    const pointer = new Vector2(10, 10)

    let plantRoot: Group | null = null
    let leavesNode: Object3D | null = null
    let leafMeshes: Mesh[] = []
    let isHovering = false

    // Smooth animation state
    let currentRotZ = 0
    let currentRotX = 0
    let targetRotZ = 0
    let targetRotX = 0
    let baseRotZ = 0
    let baseRotX = 0

    const loader = new GLTFLoader()
    loader.load(
      MODEL_PATH,
      (gltf) => {
        plantRoot = gltf.scene as Group

        // Log hierarchy for debugging
        plantRoot.traverse((child: any) => {
          const depth = getDepth(child)
          const indent = '  '.repeat(depth)
          console.log(`${indent}${child.type}: "${child.name}"`)
        })

        // Find the "Leaves" node from the model hierarchy
        // Model structure: Sketchfab_model > ... > RootNode > Leaves > Leaves_Mat.1_0
        plantRoot.traverse((child: any) => {
          const name = (child.name || '')

          // Match the "Leaves" group node exactly
          if (name === 'Leaves' && !leavesNode) {
            leavesNode = child
            console.log('Found Leaves group:', child.name, child.type)
          }

          // Collect meshes under "Leaves" for raycasting
          if (child.isMesh) {
            const path = getNodePath(child)
            if (path.includes('Leaves') || path.includes('leaves')) {
              leafMeshes.push(child)
            }
          }
        })

        // Fallback: broader search
        if (!leavesNode) {
          plantRoot.traverse((child: any) => {
            const name = (child.name || '').toLowerCase()
            if (
              (name.includes('leaves') || name.includes('leaf') || name.includes('foliage')) &&
              !leavesNode
            ) {
              leavesNode = child
            }
          })
        }

        // If no leaf meshes found for raycasting, use all meshes
        if (leafMeshes.length === 0) {
          plantRoot.traverse((child: any) => {
            if (child.isMesh) leafMeshes.push(child)
          })
        }

        console.log(`Leaves node: ${leavesNode?.name || 'NOT FOUND'}`)
        console.log(`Leaf meshes for raycasting: ${leafMeshes.length}`)

        // Store original rotation of the leaves node
        if (leavesNode) {
          baseRotZ = leavesNode.rotation.z
          baseRotX = leavesNode.rotation.x
          currentRotZ = baseRotZ
          currentRotX = baseRotX
          targetRotZ = baseRotZ
          targetRotX = baseRotX
        }

        // Auto-center and auto-scale
        const box = new Box3().setFromObject(plantRoot)
        const size = new Vector3()
        const center = new Vector3()
        box.getSize(size)
        box.getCenter(center)

        const maxDim = Math.max(size.x, size.y, size.z)
        const desiredSize = 3.2
        const scaleFactor = desiredSize / maxDim

        plantRoot.scale.setScalar(scaleFactor)
        plantRoot.position.set(
          -center.x * scaleFactor,
          -center.y * scaleFactor + desiredSize * 0.05,
          -center.z * scaleFactor
        )

        const scaledHeight = size.y * scaleFactor
        camera.position.set(0, scaledHeight * 0.45, desiredSize * 1.5)
        camera.lookAt(0, scaledHeight * 0.35, 0)

        scene.add(plantRoot)
      },
      undefined,
      (error) => {
        console.error('Error loading plant model:', error)
      }
    )

    function getDepth(obj: Object3D): number {
      let depth = 0
      let p = obj.parent
      while (p) { depth++; p = p.parent }
      return depth
    }

    function getNodePath(obj: Object3D): string {
      const parts: string[] = []
      let current: Object3D | null = obj
      while (current) {
        parts.unshift(current.name || current.type)
        current = current.parent
      }
      return parts.join(' > ')
    }

    // Resize
    const resize = () => {
      if (!container) return
      const w = container.clientWidth || 1
      const h = container.clientHeight || 1
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h, false)
    }

    resize()
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => resize()) : null
    if (ro) ro.observe(container)

    // Pointer
    const onMove = (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
    }
    const onEnter = () => { isHovering = true }
    const onLeave = () => { isHovering = false; pointer.set(10, 10) }

    renderer.domElement.addEventListener('pointermove', onMove)
    renderer.domElement.addEventListener('pointerenter', onEnter)
    renderer.domElement.addEventListener('pointerleave', onLeave)

    // Animation
    let frameId: number
    const animate = () => {
      const elapsed = clock.getElapsedTime()

      if (plantRoot && leavesNode) {
        // Plant does NOT rotate — it stays fixed in place.

        // Check hover on leaves
        let hoveringLeaf = false
        if (isHovering && leafMeshes.length > 0) {
          raycaster.setFromCamera(pointer, camera)
          const hits = raycaster.intersectObjects(leafMeshes, true)
          hoveringLeaf = hits.length > 0
        }

        // Compute target rotation for the entire Leaves group
        if (hoveringLeaf) {
          // Direct hover on leaves — strongest response
          const strength = 0.18
          const breathe = Math.sin(elapsed * 1.5) * 0.015
          targetRotZ = baseRotZ + pointer.x * strength + breathe
          targetRotX = baseRotX + pointer.y * strength * 0.5 + breathe * 0.6
        } else if (isHovering) {
          // Mouse is over the canvas but not directly on leaves — mild response
          const strength = 0.07
          const breathe = Math.sin(elapsed * 1.0) * 0.01
          targetRotZ = baseRotZ + pointer.x * strength + breathe
          targetRotX = baseRotX + pointer.y * strength * 0.4 + breathe * 0.4
        } else {
          // Mouse is off — very gentle idle breathing
          const idleAmp = 0.015
          const idleSpd = 0.5
          targetRotZ = baseRotZ + Math.sin(elapsed * idleSpd) * idleAmp
          targetRotX = baseRotX + Math.cos(elapsed * idleSpd * 0.7) * idleAmp * 0.5
        }

        // Smooth lerp — the key to buttery movement
        const lerp = 0.05
        currentRotZ += (targetRotZ - currentRotZ) * lerp
        currentRotX += (targetRotX - currentRotX) * lerp

        // Apply to the Leaves group node — stems + leaves move together
        leavesNode.rotation.z = currentRotZ
        leavesNode.rotation.x = currentRotX
      }

      renderer.render(scene, camera)
      frameId = requestAnimationFrame(animate)
    }

    frameId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(frameId)
      renderer.domElement.removeEventListener('pointermove', onMove)
      renderer.domElement.removeEventListener('pointerenter', onEnter)
      renderer.domElement.removeEventListener('pointerleave', onLeave)
      if (ro && container) ro.disconnect()
      if (plantRoot) scene.remove(plantRoot)
      renderer.dispose()
      if (renderer.domElement.parentElement) {
        renderer.domElement.parentElement.removeChild(renderer.domElement)
      }
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className="w-full h-full relative"
      style={{ minHeight: '420px', cursor: 'grab' }}
    />
  )
}