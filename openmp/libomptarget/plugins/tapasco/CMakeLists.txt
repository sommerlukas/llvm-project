##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a TaPaSCo machine if available.
#
##===----------------------------------------------------------------------===##
if (NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building TaPaSCo offloading plugin: only support TaPaSCo in Linux x86_64 hosts.")
  return()
elseif (NOT LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building TaPaSCo offloading plugin: libelf dependency not found.")
  return()
elseif (NOT LIBOMPTARGET_DEP_LIBFFI_FOUND)
  libomptarget_say("Not building TaPaSCo offloading plugin: libffi dependency not found.")
  return()
elseif(NOT LIBOMPTARGET_DEP_TAPASCO_FOUND)
  libomptarget_say("Not building TaPaSCo offloading plugin: TaPaSCo not found in system.")
  return()
endif()

libomptarget_say("Building TaPaSCo offloading plugin.")

# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=TAPASCO)

add_library(omptarget.rtl.tapasco SHARED src/rtl.cpp)

install(TARGETS omptarget.rtl.tapasco LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")

target_link_libraries(omptarget.rtl.tapasco
	tapasco
	platform
	ffi
	dl
	elf
	"-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")

# Report to the parent scope that we are building a plugin for TaPaSCo.
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} tapasco-x86_64-linux" PARENT_SCOPE)
