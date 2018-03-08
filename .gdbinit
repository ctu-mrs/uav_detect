python
import sys
sys.path.insert(0, '/home/mrs/.gdb/eigen/')
from printers import register_eigen_printers
register_eigen_printers (None)
end

add-auto-load-safe-path /home/mrs/git/darknet_ocl/darknet_ros/.gdbinit
set breakpoint pending on
break detected_uav.cpp:80
# break detected_uav.cpp:268
# break detected_uav.cpp:248
