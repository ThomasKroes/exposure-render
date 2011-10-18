This application implements the concepts and ideas expressed in the paper: "Direct Volume Rendering with Physically Based Lighting".

Work flow tutorials (video's) will be put online as soon as possible.

For a head start you can open one of the demo files, which come with appearance, lighting and camera presets.

=== Operating the camera ===
- LMB to orbit
- MMB to pan
- RMB + mouse wheel to zoom
- Shift + LMB to adjust the aperture size
- Control + LMB to adjust the field of view
- Alt + LMB to adjust the focal distance

Note: The auto focus point is located in the center of the render canvas, future versions will include clickable focus points

=== Navigating the render canvas ===
- SPACE + MMB to pan
- SPACE + RMB to zoom
- SPACE + mouse wheel to zoom

=== Editing the transfer function ===
- Left click in empty areas to create a new node
- RMB on a node to delete it

=== System Requirements ===
- Microsoft Windows XP, Vista, or 7
- NVIDIA CUDA-compatible GPU with compute capability 1.0+ and at least 512 megabytes of DRAM. GeForce GTX2XX is recommended
- At least 1GB of system memory

=== Changes ===
- Overall performance improvements
- GUI improvements
- Noise reduction

=== Future Plans ===
- Performance optimizations
- OpenCL support
- HDRI textured lights
- Specular Bloom
- Clipping
- Mesh based lights
- More advanced 2D sampling schemes
- Turntable animation rendering

=== Known Limitations ===
- Large volumes can cause Exposure Render to crash, we are working on this