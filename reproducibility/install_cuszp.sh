#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="https://github.com/szcompressor/cuSZp.git"
COMMIT="f581dcf329c907c320f4743a9c6e7ee2fb9c5494"
TARGET="${CUSZP_ROOT:-/opt/cuSZp}"
CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90}"

if [[ -e "${TARGET}" ]]; then
  if [[ ! -d "${TARGET}/.git" ]]; then
    echo "Refusing to replace non-Git path: ${TARGET}" >&2
    exit 1
  fi
  actual="$(git -C "${TARGET}" rev-parse HEAD)"
  if [[ "${actual}" != "${COMMIT}" ]]; then
    echo "Existing cuSZp checkout is at ${actual}, expected ${COMMIT}." >&2
    echo "Use a new CUSZP_ROOT or update that checkout explicitly." >&2
    exit 1
  fi
else
  git clone "${REPOSITORY}" "${TARGET}"
  git -C "${TARGET}" checkout --detach "${COMMIT}"
fi

cmake -S "${TARGET}" -B "${TARGET}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${TARGET}/install" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DcuSZp_BUILD_EXAMPLES=OFF
cmake --build "${TARGET}/build" -j"$(nproc)"
cmake --install "${TARGET}/build"

echo "Pinned cuSZp installed under ${TARGET}/install"
