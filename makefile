compile_metal_shaders:
	xcrun -sdk macosx metal -c src/tensor/tensor_ops/${SHADER}/metal/${SHADER}.metal -o src/tensor/tensor_ops/${SHADER}/metal/${SHADER}.air
	xcrun -sdk macosx metallib src/tensor/tensor_ops/${SHADER}/metal/${SHADER}.air -o src/tensor/tensor_ops/${SHADER}/metal/${SHADER}.metallib