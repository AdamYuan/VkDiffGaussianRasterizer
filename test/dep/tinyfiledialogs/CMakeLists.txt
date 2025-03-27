cmake_minimum_required(VERSION 3.15)
project(tinyfiledialogs C)

add_library(tinyfiledialogs STATIC tinyfiledialogs.c)
target_include_directories(tinyfiledialogs INTERFACE .)