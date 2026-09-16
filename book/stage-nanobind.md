# Stage 3: nanobind, the kernel compiled ahead of time

Stage 2 is a perfectly reasonable place to stop.
A package whose kernel is compiled by NVRTC at first call is pure Python, builds on any machine, and runs the same kernel at the same speed as anything this book goes on to build.
For a project used by its own authors, on machines they control, that is often the right trade.

Stage 3 is the alternative for a package meant for other people.
It takes the kernel from stage 2 unchanged, wraps it in a nanobind binding, and compiles it with `nvcc` when the package is built rather than when it is used.
The `gpu-pairwise` package that results is platform specific and tied to one CPython ABI, and the workspace has to pin a CUDA compiler, so there is more to maintain.
What that buys is worth spelling out, because it is the whole reason to go further.

## When it is worth it

- **Errors surface on the author's machine.** A kernel that does not compile fails the package build, not a user's first call.
- **Nothing compiles at runtime.** Users need the CUDA runtime library and a driver, not NVRTC, and the first call is as fast as every other.
- **One binary serves every GPU generation.** `nvcc` embeds device code for every major architecture, where NVRTC targets only the GPU that happens to be present.
- **The metadata writes itself.** The package's CUDA runtime constraint is derived from conda-forge run-exports rather than typed by hand, so it cannot be forgotten or drift out of date.

If none of those matter for your project, stage 2 is the better stage and the extra machinery below is cost without benefit.
If any of them do, the following chapters show that the machinery is smaller than it looks: one CMake file, one binding file, and a handful of manifest lines.

## The chapters

Stage 3 is walked from the outside in.
[The workspace](./workspace.md) is what you `pixi run`.
[The package](./package.md) is what Pixi turns into a `.conda` file.
[The kernel and its bindings](./kernel.md) are the code being packaged.
[Running it](./running.md) shows the build, the demo, the tests, and the benchmark.
[Distributing it](./distributing.md) publishes the package to a local channel, reads its metadata back, and installs it from a fresh workspace.
