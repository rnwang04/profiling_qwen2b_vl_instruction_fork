/home/arda/ruonan/openvino/build/install/setupvars.sh
export OpenVINO_DIR=/home/arda/ruonan/openvino/build
export OpenCV_DIR=/home/arda/ruonan/opencv/build
export OpenVINOGenAI_DIR=/home/arda/ruonan/openvino.genai/build

mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j32
