# CMake generated Testfile for 
# Source directory: /home/manab/colcon_ws/src/ur_description
# Build directory: /home/manab/colcon_ws/src/ur5_control/build/ur_description
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ur_urdf_xacro "/usr/bin/python3" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/home/manab/colcon_ws/src/ur5_control/build/ur_description/test_results/ur_description/ur_urdf_xacro.xunit.xml" "--package-name" "ur_description" "--output-file" "/home/manab/colcon_ws/src/ur5_control/build/ur_description/ament_cmake_pytest/ur_urdf_xacro.txt" "--command" "/usr/bin/python3" "-u" "-m" "pytest" "/home/manab/colcon_ws/src/ur_description/test/test_ur_urdf_xacro.py" "-o" "cache_dir=/home/manab/colcon_ws/src/ur5_control/build/ur_description/ament_cmake_pytest/ur_urdf_xacro/.cache" "--junit-xml=/home/manab/colcon_ws/src/ur5_control/build/ur_description/test_results/ur_description/ur_urdf_xacro.xunit.xml" "--junit-prefix=ur_description")
set_tests_properties(ur_urdf_xacro PROPERTIES  LABELS "pytest" TIMEOUT "60" WORKING_DIRECTORY "/home/manab/colcon_ws/src/ur5_control/build/ur_description" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/ament_cmake_pytest/cmake/ament_add_pytest_test.cmake;169;ament_add_test;/home/manab/colcon_ws/src/ur_description/CMakeLists.txt;18;ament_add_pytest_test;/home/manab/colcon_ws/src/ur_description/CMakeLists.txt;0;")
add_test(view_ur_launch "/usr/bin/python3" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/home/manab/colcon_ws/src/ur5_control/build/ur_description/test_results/ur_description/view_ur_launch.xunit.xml" "--package-name" "ur_description" "--output-file" "/home/manab/colcon_ws/src/ur5_control/build/ur_description/ament_cmake_pytest/view_ur_launch.txt" "--command" "/usr/bin/python3" "-u" "-m" "pytest" "/home/manab/colcon_ws/src/ur_description/test/test_view_ur_launch.py" "-o" "cache_dir=/home/manab/colcon_ws/src/ur5_control/build/ur_description/ament_cmake_pytest/view_ur_launch/.cache" "--junit-xml=/home/manab/colcon_ws/src/ur5_control/build/ur_description/test_results/ur_description/view_ur_launch.xunit.xml" "--junit-prefix=ur_description")
set_tests_properties(view_ur_launch PROPERTIES  LABELS "pytest" TIMEOUT "60" WORKING_DIRECTORY "/home/manab/colcon_ws/src/ur5_control/build/ur_description" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/ament_cmake_pytest/cmake/ament_add_pytest_test.cmake;169;ament_add_test;/home/manab/colcon_ws/src/ur_description/CMakeLists.txt;19;ament_add_pytest_test;/home/manab/colcon_ws/src/ur_description/CMakeLists.txt;0;")
