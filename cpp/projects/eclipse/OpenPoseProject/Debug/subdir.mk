################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../AutoResetEvent.cpp \
../ClassCamCalib.cpp \
../ClassMJPEGReader.cpp \
../ClassMain.cpp \
../ClassOpenPose.cpp \
../ClassPoseResults.cpp \
../ClassTest.cpp \
../ClassWrapper.cpp \
../FileHandler.cpp \
../FrameInfo.cpp \
../ImageProcess.cpp \
../VideoCapInfo.cpp \
../main.cpp 

OBJS += \
./AutoResetEvent.o \
./ClassCamCalib.o \
./ClassMJPEGReader.o \
./ClassMain.o \
./ClassOpenPose.o \
./ClassPoseResults.o \
./ClassTest.o \
./ClassWrapper.o \
./FileHandler.o \
./FrameInfo.o \
./ImageProcess.o \
./VideoCapInfo.o \
./main.o 

CPP_DEPS += \
./AutoResetEvent.d \
./ClassCamCalib.d \
./ClassMJPEGReader.d \
./ClassMain.d \
./ClassOpenPose.d \
./ClassPoseResults.d \
./ClassTest.d \
./ClassWrapper.d \
./FileHandler.d \
./FrameInfo.d \
./ImageProcess.d \
./VideoCapInfo.d \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -I/usr/lib/jvm/java-11-openjdk-amd64/include/ -I/usr/lib/jvm/java-11-openjdk-amd64/include/linux -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


