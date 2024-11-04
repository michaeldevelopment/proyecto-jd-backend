const fs = require("fs");
const csv = require("csv-parser");
const tf = require("@tensorflow/tfjs");

const input = [];
const output = [];

function loadCSVData() {
  let index = 0;
  return new Promise((resolve, reject) => {
    fs.createReadStream("./data/data.csv")
      .pipe(csv({ separator: ";" }))
      .on("data", (row) => {
        input[index] = [
          Number(row.precipitacion),
          Number(row.temperatura),
          Number(row.irradiancia),
          Number(row.area),
          Number(row.alfa),
          Number(row.numeroDePaneles),
        ];
        output[index] = Number(row.potencia);
        index++;
      })
      .on("end", async () => {
        resolve();
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}

async function predict(prec, temp, irr, numPaneles) {
  await loadCSVData();

  const xs = tf.tensor2d(input);
  const ys = tf.tensor2d(output, [output.length, 1]);
  const fixArea = 150.47;
  const fixAlfa = -0.0035;

  // Define el modelo
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 64, inputShape: [6], activation: "relu" })
  ); // 64 neuronas en la capa oculta
  model.add(tf.layers.dense({ units: 32, activation: "relu" })); // Segunda capa oculta con 32 neuronas
  model.add(tf.layers.dense({ units: 1 })); // Capa de salida para la estimación de potencia

  // Compila el modelo con la función de pérdida y el optimizador
  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(0.001), // Reducir la tasa de aprendizaje
  });

  // Entrena el modelo
  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
  });

  xs.dispose();
  ys.dispose();

  // Ejemplo de predicción
  // precipitacion, temperatura, irradiancia, area, alfa, numero de paneles.
  // const sampleInput = tf.tensor2d([[0.07, 22, 20, fixArea, fixAlfa, 2]]);
  const sampleInput = tf.tensor2d([
    [prec, temp, irr, fixArea, fixAlfa, numPaneles],
  ]);
  const prediction = model.predict(sampleInput);
  return prediction.dataSync()[0];
}

module.exports = predict;
