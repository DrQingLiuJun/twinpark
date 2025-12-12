cmake_minimum_required(VERSION 3.0.2)
project(twinpark_launch)

## Find catkin macros and libraries
find_package(catkin REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package()

#############
## Install ##
#############

## Mark launch files for installation
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Mark config files for installation
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

## Mark scripts for installation
install(PROGRAMS
  scripts/visualization_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
