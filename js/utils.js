function stringToDOMElement(htmlContent){
	var tmpElement = document.createElement('div')
	tmpElement.innerHTML = htmlContent.trim()
	var domElement = tmpElement.firstChild  
	return domElement                      
}        
function drawImageTensor(imageTensor, canvas) {
	const [width, height] = [224, 224]
	const context = canvas.getContext('2d')
	const imageData = new ImageData(width, height)
	const data = imageTensor.dataSync()
	for (let i = 0; i < height * width; ++i) {
		const j = i * 4
		imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127
		imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127
		imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127
		imageData.data[j + 3] = 255
	}
	context.putImageData(imageData, 0, 0)
}
