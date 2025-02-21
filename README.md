# husky-assembly

## Note

* 如果使用WSL，在Rviz中不能正常显示mesh，可以采取下面的方案：
  * 方案1：```export LIBGL_ALWAYS_SOFTWARE=1```
  * 方案2：[参考链接](https://blog.csdn.net/GodNotAMen/article/details/125123186)
* urdf无法正常加载：```export ROS_PACKAGE_PATH=<path_to_husky_assembly>/data/husky_urdf:$ROS_PACKAGE_PATH```
* 报错```ImportError: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0```：
  * 降级python到3.8.10
  * ```conda install libffi==3.3```