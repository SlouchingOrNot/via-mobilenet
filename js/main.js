(async function(){
        
	//////////////////////////////////////////////////////////////////////////////
	//                Code Separator
	//////////////////////////////////////////////////////////////////////////////
        
	var CLASSNAMES = [
		'good',
		'bad',
		'ignore',
	]
	var NUM_CLASSES = CLASSNAMES.length
	controllerDataset = new ControllerDataset(NUM_CLASSES)
        
	var model = null
        
	//////////////////////////////////////////////////////////////////////////////
	//                predicting
	//////////////////////////////////////////////////////////////////////////////
	var isPredicting = false
	async function startPredictingLoop() {
		isPredicting = true
		document.querySelector('#ui-prediction-start').innerText = 'Stop'
                
		console.log('isPredicting')
		while (isPredicting) {
			const predictions = tf.tidy(() => {
				// Capture the frame from the webcam.
				const img = webcam.capture()
                                
				// Make a prediction through mobilenet, getting the internal activation of
				// the mobilenet model.
				const activation = mobilenet.predict(img)
                                
				// Make a prediction through our newly-trained model using the activation
				// from mobilenet as input.
				const predictions = model.predict(activation)
                                
				return predictions
			})
                        
                        
			var predictionsData = await predictions.data()                        
			predictions.dispose()
                        
			//////////////////////////////////////////////////////////////////////////////
			//                update UI
			//////////////////////////////////////////////////////////////////////////////
			var bestLabel = null
			for(let label = 0; label < predictionsData.length; label++){
				if( bestLabel === null || predictionsData[label] > predictionsData[bestLabel]){
					bestLabel = label
				}
			}
                        
			document.querySelector('#ui-prediction-bestclass').innerText = CLASSNAMES[bestLabel]
                        
			for(let label = 0; label < predictionsData.length; label++){
				var domElement = document.querySelector(`#ui-prediction-${CLASSNAMES[label]}`)
				domElement.innerHTML = (predictionsData[label]*100).toFixed(2)
				if( label === bestLabel ){
					domElement.style.color = 'darkred'
					domElement.style.fontWeight = 'bold'
				}else{
					domElement.style.color = ''                                        
					domElement.style.fontWeight = ''
				}
			}
                        
			//////////////////////////////////////////////////////////////////////////////
			//                wait for next frame
			//////////////////////////////////////////////////////////////////////////////
			await tf.nextFrame()
		}
		console.log('donePredicting')
		document.querySelector('#ui-prediction-start').innerText = 'Start'
	}        
        
	var domElement = document.querySelector('#ui-prediction-start')
	domElement.disabled = 'disabled'
	domElement.addEventListener('click', function(){
		if( isPredicting === false ){
			startPredictingLoop()
		}else{
			isPredicting = false                        
		}
		updateAppStateUI()
	})
        
        
	//////////////////////////////////////////////////////////////////////////////
	//                Model IO
	//////////////////////////////////////////////////////////////////////////////
        
	class ModelIO {
		static async save(){
			var modelURL = 'indexeddb://my-model-1'
			const saveResult = await model.save(modelURL)
                        
			await modelIOUpdateUI()
		}
		static async load(){
			var modelURL = 'indexeddb://my-model-1'
			model = await tf.loadModel(modelURL)
			console.log(`loadModel=${model}`)
                        
			document.querySelector('#ui-prediction-start').disabled = ''
                        
			await modelIOUpdateUI()                        
		}
		static async delete(){
			var modelURL = 'indexeddb://my-model-1'
			const result = await tf.io.removeModel(modelURL)
                        
			await modelIOUpdateUI()                    
		}
		static async exists(){
			var modelURL = 'indexeddb://my-model-1'
			var listModels = await tf.io.listModels()
			var result = Object.keys(listModels).indexOf(modelURL)
			var found = result !== -1 ? true : false
			return found                        
		}
	}
        
	// save button
	document.querySelector('#ui-model-save').addEventListener('click', () => ModelIO.save())
	document.querySelector('#ui-model-load').addEventListener('click', () => ModelIO.load())
	document.querySelector('#ui-model-delete').addEventListener('click', () => ModelIO.delete())
        
	// util function
	async function modelIOUpdateUI(){
		var hasStoredModel = (await ModelIO.exists()) === true ? true : false
                
		document.querySelector('#ui-model-load').disabled = hasStoredModel ? '' : 'disabled'
		document.querySelector('#ui-model-delete').disabled = hasStoredModel ? '' : 'disabled'
		document.querySelector('#ui-model-stored').innerHTML = hasStoredModel ? '<span style="color:green">yes</span>' : 'no'
                
		var hasTrainedModel = model !== null ? true : false
		console.log('hasTrainedModel', hasTrainedModel)
		document.querySelector('#ui-model-save').disabled = hasTrainedModel ? '' : 'disabled'
	}
        
	await modelIOUpdateUI()
        
	//////////////////////////////////////////////////////////////////////////////
	//                data collection
	//////////////////////////////////////////////////////////////////////////////
        
	CLASSNAMES.forEach(function(className, classIndex){
		var domElement = document.querySelector(`#ui-data-collection-${className} button`)
                
		var canvasEl = document.querySelector(`#ui-data-collection-${className} canvas`)
		var context = canvasEl.getContext('2d')
		context.fillRect(0,0, canvasEl.width, canvasEl.height)
                
		// add examples continuously
		domElement.addEventListener('mousedown', function(){
			var addingExamples = true
                        
			domElement.addEventListener('mouseup', function callback(){
				domElement.removeEventListener('mouseup', callback)
				addingExamples = false
			})
			domElement.addEventListener('mouseleave', function callback(){
				domElement.removeEventListener('mouseleave', callback)
				addingExamples = false
			})
                        
			tick()
			return
                        
			function tick(){
				// controllerDataset
				const imageTensor = webcam.capture()
				const activations = mobilenet.predict(imageTensor)
				controllerDataset.addExample(activations, classIndex)
                                
				var canvasEl = document.querySelector(`#ui-data-collection-${className} canvas`)
				drawImageTensor(imageTensor, canvasEl)
                                
				var numExampleEl = document.querySelector(`#ui-data-collection-${className} .numExamples`)
				var numExamples = parseInt(numExampleEl.innerHTML)+1
				numExampleEl.innerHTML = numExamples
                                
				if( addingExamples === true ){
					setTimeout(tick, 1000/10)
					return
				}
			}
		})                        
	})
        
	//////////////////////////////////////////////////////////////////////////////
	//                Training
	//////////////////////////////////////////////////////////////////////////////
        
	var isTraining = false
	document.querySelector('#ui-data-collection-train').addEventListener('click', async function(){
                
		// sanity check
		if (controllerDataset.xs == null) {
			// display toast to notify the user
			document.querySelector('#demo-toast-example').MaterialSnackbar.showSnackbar({
				message: 'Add Samples Before Training'
			})
			return null
		}
                
                
		document.querySelector('#ui-data-collection-train').disabled = 'disabled'
		document.querySelector('#ui-prediction-start').disabled = 'disabled'
                
		isPredicting = false
		isTraining = true
                
		updateAppStateUI()
                
		setTimeout(async function(){
			model = await buildAndTrainModel(controllerDataset)
			document.querySelector('#ui-data-collection-train').disabled = ''
			document.querySelector('#ui-prediction-start').disabled = ''
                        
			await modelIOUpdateUI()
                        
			if( await ModelIO.exists() === false ){
				await ModelIO.save()
			}
                        
			isTraining = false
                        
			updateAppStateUI()
                        
			startPredictingLoop()
		}, 100)
                
	})
        
	document.querySelector('#ui-data-collection-clear').addEventListener('click', async function(){
		controllerDataset.clear()
                
		document.querySelector('#ui-data-collection-epoch-index').innerText = 0
		document.querySelector('#ui-data-collection-loss').innerText = 0
                
                
		CLASSNAMES.forEach(function(className){
			document.querySelector(`#ui-data-collection-${className} .numExamples`).innerText = 0
		})
	})
        
	/**
        * Sets up and trains the classifier.
        */
	async function buildAndTrainModel(controllerDataset) {
                
		var ui = {
			getDenseUnits: () => 100,
			getLearningRate: () => 0.0001,
			getBatchSizeFraction: () =>  0.4,
			getEpochs: () => 20,
		}
                
		// Creates a 2-layer fully connected model. By creating a separate model,
		// rather than adding layers to the mobilenet model, we "freeze" the weights
		// of the mobilenet model, and only train weights from the new model.
		var model = tf.sequential({
			layers: [
				// Flattens the input to a vector so we can use it in a dense layer. While
				// technically a layer, this only performs a reshape (and has no training
				// parameters).
				tf.layers.flatten({inputShape: [7, 7, 256]}),
				// Layer 1
				tf.layers.dense({
					units: ui.getDenseUnits(),
					activation: 'relu',
					kernelInitializer: 'varianceScaling',
					useBias: true
				}),
				// Layer 2. The number of units of the last layer should correspond
				// to the number of classes we want to predict.
				tf.layers.dense({
					units: NUM_CLASSES,
					activation: 'softmax',
					kernelInitializer: 'varianceScaling',
					useBias: false,
				})
			]
		})
		// window.model = model
                
		// Creates the optimizers which drives training of the model.
		const optimizer = tf.train.adam(ui.getLearningRate())
		// We use categoricalCrossentropy which is the loss function we use for
		// categorical classification which measures the error between our predicted
		// probability distribution over classes (probability that an input is of each
		// class), versus the label (100% probability in the true class)>
		model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'})
                
		// We parameterize batch size as a fraction of the entire dataset because the
		// number of examples that are collected depends on how many examples the user
		// collects. This allows us to have a flexible batch size.
		const batchSize = Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction())
		if (!(batchSize > 0)) {
			throw new Error(
				'Batch size is 0 or NaN. Please choose a non-zero fraction.'
			)
		}
                
		document.querySelector('#ui-data-collection-epoch-index').innerText = 0
		document.querySelector('#ui-data-collection-epoch-count').innerText = ui.getEpochs()
                
		// Train the model! Model.fit() will shuffle xs & ys so we don't have to.
		await model.fit(controllerDataset.xs, controllerDataset.ys, {
			batchSize,
			epochs: ui.getEpochs(),
			callbacks: {
				onEpochEnd: () => {
					var domElement = document.querySelector('#ui-data-collection-epoch-index')
					var epoch = parseInt(domElement.innerText.trim(), 10)
					epoch += 1
					domElement.innerText = epoch
				},
				onBatchEnd: async (batch, logs) => {
					document.querySelector('#ui-data-collection-loss').innerText = logs.loss.toFixed(5)
				}
			}
		})
                
		console.log('training complete')
		return model
	}
        
        
	//////////////////////////////////////////////////////////////////////////////
	//                Init Video
	//////////////////////////////////////////////////////////////////////////////
	// A webcam class that generates Tensors from the images from the webcam.
        
	var videoEl = document.querySelector('#webcam')
	videoEl.style.transform = 'scaleX(-1)'
        
	var webcam = new Webcam(videoEl)                
	await webcam.setup()
        
	//////////////////////////////////////////////////////////////////////////////
	//                load mobilenet
	//////////////////////////////////////////////////////////////////////////////
	// async function loadMobilenet() {
	// 	const mobilenet = await tf.loadModel(
	// 		'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
	// 	)
	// 
	// 	// Return a model that outputs an internal activation.
	// 	const layer = mobilenet.getLayer('conv_pw_13_relu')
	// 	return tf.model({inputs: mobilenet.inputs, outputs: layer.output})
	// }
	class MobileNetTrunkated {
		// Loads mobilenet and returns a model that returns the internal activation
		// we'll use as input to our classifier model.
		static async loadAndCreate() {
			const mobilenet = await tf.loadModel(
				'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
			)
                        
			// Return a model that outputs an internal activation.
			const layer = mobilenet.getLayer('conv_pw_13_relu')
			return tf.model({inputs: mobilenet.inputs, outputs: layer.output})
		}
		static warmUp(mobilenet){
			// Warm up the model. This uploads weights to the GPU and compiles the WebGL
			// programs so the first time we collect data from the webcam it will be
			// quick.
			tf.tidy(() => mobilenet.predict(webcam.capture()))                        
		}
	}
        
	// var mobilenet = await loadMobilenet()
	var mobilenet = await MobileNetTrunkated.loadAndCreate()
        
	MobileNetTrunkated.warmUp(mobilenet)
        
	// // Warm up the model. This uploads weights to the GPU and compiles the WebGL
	// // programs so the first time we collect data from the webcam it will be
	// // quick.
	// tf.tidy(() => mobilenet.predict(webcam.capture()))
        
	// display toast to notify the user
	document.querySelector('#demo-toast-example').MaterialSnackbar.showSnackbar({
		message: 'Model Loaded'
	})
        
        
	// if model stored, load it and start predicting once loaded
	if( await ModelIO.exists() === true ){
		await ModelIO.load()
		startPredictingLoop()
	}
        
	//////////////////////////////////////////////////////////////////////////////
	//                app state
	//////////////////////////////////////////////////////////////////////////////
        
	function updateAppStateUI(){
		var hasLoadedModel = mobilenet !== undefined ? true : false
		if( isPredicting === true ){
			document.querySelector('#ui-app-state').innerHTML = 'Predicting'
		}else if( isTraining === true ){
			document.querySelector('#ui-app-state').innerHTML = 'Training'
		}else if( hasLoadedModel === true ){
			document.querySelector('#ui-app-state').innerHTML = 'Ready'
		}else{
			document.querySelector('#ui-app-state').innerHTML = 'Loading'
		}
	}
        
	updateAppStateUI()
        
	// var htmlContent = `
	//         <div>
	//                 <div class="mdl-spinner mdl-spinner--single-color mdl-js-spinner is-active"></div>'
	//                 Loading
	//         </div>
	// `
	// var domElement = createDomElement(htmlContent)
	// document.querySelector('.demo-content').appendChild(domElement);
	// 
	// function createDomElement(htmlcontent){
	//         var domElement = document.createElement('div');
	//         domElement.innerHTML =  htmlContent;
	//         return domElement.firstChild
	// }
        
        
        
})()
