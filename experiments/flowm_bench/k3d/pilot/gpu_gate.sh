# source this; wait_for_gpu_free <MiB> [gpu index] blocks until that much memory is free on the GPU (default 1).
wait_for_gpu_free() {
  local need=$1 gpu=${2:-1} free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
    [ -n "$free" ] && [ "$free" -ge "$need" ] && return 0
    sleep 300
  done
}
