cmd_/root/c-workspace/cgpu/modules.order := {   echo /root/c-workspace/cgpu/cgpu_km.ko; :; } | awk '!x[$$0]++' - > /root/c-workspace/cgpu/modules.order
