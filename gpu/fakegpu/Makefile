obj-m += fake_gpu.o
all:
 	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
test:
  # We put a — in front of the rmmod command to tell make to ignore
  # an error in case the module isn’t loaded.
  -sudo rmmod fake_gpu
  # Clear the kernel log without echo
  sudo dmesg -C
  # Insert the module
  sudo insmod fake_gpu.ko
  # Display the kernel log
  dmesg

run:
	docker run -ti -e NVIDIA_VISIBLE_DEVICES=none --device /dev/fake_gpu0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04 /bin/bash