const tf = require('@tensorflow/tfjs');
const iris = require('./iris.json');
const irisTesting = require('./testingIris.json');

// convert data
const trainingData = tf.tensor2d(iris.map(item=> [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]
),[130,4])

const outputData = tf.tensor2d(iris.map(item => [
    item.species === 'setosa' ? 1 : 0,
    item.species === 'virginica' ? 1 : 0,
    item.species === 'versicolor' ? 1 : 0

]), [130,3])

// creating model
const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [4], //we've four features that's why
    activation: "sigmoid",
    units: 10
}))

model.add(tf.layers.dense({
    inputShape: [10], //we've four features that's why
    activation: "softmax",
    units: 3
}));

model.summary();
// compiling model
model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam()
})

async function train_data(){
    for(let i=0;i<15;i++){
       const res = await model.fit(trainingData,
                   outputData,{epochs: 40});  
       console.log(res.history.loss[0]);          
    }
 }

 async function main() {
    let train = await train_data();
    console.log('Model Prediction')
    model.predict(estingData).print();
  }

// predicting model
main();