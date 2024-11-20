const fs = require("fs");
const csv = require("csv-parser");
const tf = require("@tensorflow/tfjs");

const input = [];
const output = [];

function loadCSVData() {
  let index = 0;
  return new Promise((resolve, reject) => {
    fs.createReadStream("./data/nuevaData.csv")
      .pipe(csv({ separator: ";" }))
      .on("data", (row) => {
        input[index] = [
          Number(row.hora),
          Number(row.precipitacion),
          Number(row.temperatura),
        ];
        output[index] = [Number(row.irradiancia)];
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

async function predict(data) {
  await loadCSVData();

  const inputArray = data.map((data) => [
    data.hora,
    data.precipitacion,
    data.temperatura,
  ]);

  const xs = tf.tensor2d(input);
  const ys = tf.tensor2d(output);

  // Define el modelo
  const model = tf.sequential();

  // Capa oculta con menor cantidad de neuronas
  model.add(
    tf.layers.dense({ units: 16, activation: "relu", inputShape: [3] })
  );

  // Capa de salida
  model.add(tf.layers.dense({ units: 1, activation: "linear" }));

  // Compilación del modelo con una tasa de aprendizaje mayor para acelerar el entrenamiento
  const learningRate = 0.01; // tasa de aprendizaje más alta
  const optimizer = tf.train.adam(learningRate);
  model.compile({
    optimizer: optimizer,
    loss: "meanSquaredError",
  });

  await model.fit(xs, ys, {
    epochs: 20, // Aumenta las épocas si los resultados siguen siendo imprecisos
    batchSize: 32,
    validationSplit: 0.2, // Usa un 20% para validación
  });

  xs.dispose();
  ys.dispose();

  const sampleInput = tf.tensor2d(inputArray);
  const prediction = model.predict(sampleInput);
  return prediction.arraySync().flat();
}

module.exports = predict;
