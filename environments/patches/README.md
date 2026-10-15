These patches preserve the native HT checkpoint support used by the evaluation adapters.

- `lerobot-pi05.patch`: LeRobot commit `a16f34c085c9597fcbdb9fde395a3334d78df716`; Apache-2.0, see `LICENSE.pi05`.
- `cosmos-framework.patch`: Cosmos Framework commit `18d90eacbda778e666e6978dc1ab6ba8635400e3`; OpenMDW-1.1, see `LICENSE.cosmos-framework`.
- `gr00t-install.patch`: GR00T commit `b861ec9090b841cc2bbae3cdc6b8d2a384a3660b`; limits this release's environment to Linux x86_64 and removes a reference to an unpublished ARM wheel. See `LICENSE.gr00t`.
- `robosuite-python310.patch`: Diffusion Policy's robosuite fork at `277ab9588ad7a4f4b55cf75508b44aa67ec171f0`; the same two `collections.abc` fixes used by Praxis on Python 3.10. MIT, see `LICENSE.robosuite`.

`scripts/install.py` fetches these exact source revisions and applies each patch once. GR00T already includes HT support in its pinned source revision.

RoboMimic uses robomimic 0.2.0, free-mujoco-py 2.1.6 with its bundled MuJoCo 2.1.0, and numba 0.56.4, matching `Praxis/scripts/cluster/bootstrap_dp.sh`. `robomimic-runtime.json` pins the Ubuntu 22.04 OSMesa 23.2.1 and GLFW 3.3.6 packages used by Praxis's verified simulator environment, plus OpenGL build headers/libraries. The installer verifies checksums and extracts these packages locally; it does not alter system packages. A C compiler, `dpkg-deb`, and the system dependencies of OSMesa (including `libllvm15`) must be available.
