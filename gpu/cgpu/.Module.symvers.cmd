cmd_/root/c-workspace/cgpu/Module.symvers := sed 's/\.ko$$/\.o/' /root/c-workspace/cgpu/modules.order | scripts/mod/modpost -m -a  -o /root/c-workspace/cgpu/Module.symvers -e -i Module.symvers   -T -
