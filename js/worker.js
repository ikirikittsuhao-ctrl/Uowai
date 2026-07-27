self.onmessage = function(e) {
  const { textData, totalEpochs } = e.data;

  // 空白区切りの単語/トークンリストを作成
  const tokens = textData.trim().split(/\s+/).filter(t => t.length > 0);
  const uniqueChars = Array.from(new Set(tokens));
  const vocabSize = uniqueChars.length;
  
  const charToId = {};
  const idToChar = {};
  uniqueChars.forEach((char, index) => {
    charToId[char] = index;
    idToChar[index] = char;
  });

  const sequenceLength = 3;  
  const hiddenUnits = 64;    
  const learningRate = 0.05; 

  const inputVectorSize = sequenceLength * vocabSize;

  let weightsInputToHidden = [];
  for (let i = 0; i < inputVectorSize; i++) {
    let row = [];
    for (let j = 0; j < hiddenUnits; j++) {
      row.push((Math.random() - 0.5) * 0.1);
    }
    weightsInputToHidden.push(row);
  }

  let weightsHiddenToOutput = [];
  for (let i = 0; i < hiddenUnits; i++) {
    let row = [];
    for (let j = 0; j < vocabSize; j++) {
      row.push((Math.random() - 0.5) * 0.1);
    }
    weightsHiddenToOutput.push(row);
  }

  const dataset = [];
  for (let i = 0; i < tokens.length - sequenceLength; i++) {
    const inputChars = tokens.slice(i, i + sequenceLength);
    const targetChar = tokens[i + sequenceLength];
    
    const inputIds = inputChars.map(c => charToId[c]);
    const targetId = charToId[targetChar];
    
    if (inputIds.every(id => id !== undefined) && targetId !== undefined) {
      dataset.push({ inputIds, targetId });
    }
  }

  if (dataset.length === 0) {
    self.postMessage({
      type: 'error',
      message: '学習データ数が不足しています。単語数を増やしてください。'
    });
    return;
  }

  for (let epoch = 0; epoch < totalEpochs; epoch++) {
    let totalLoss = 0;

    for (let sampleIndex = 0; sampleIndex < dataset.length; sampleIndex++) {
      const { inputIds, targetId } = dataset[sampleIndex];

      const xVector = new Array(inputVectorSize).fill(0);
      inputIds.forEach((id, pos) => {
        xVector[pos * vocabSize + id] = 1;
      });

      const hiddenLayer = new Array(hiddenUnits).fill(0);
      for (let h = 0; h < hiddenUnits; h++) {
        let sum = 0;
        for (let inIdx = 0; inIdx < inputVectorSize; inIdx++) {
          sum += xVector[inIdx] * weightsInputToHidden[inIdx][h];
        }
        hiddenLayer[h] = Math.max(0, sum);
      }

      const rawOutputs = new Array(vocabSize).fill(0);
      for (let outIdx = 0; outIdx < vocabSize; outIdx++) {
        let sum = 0;
        for (let h = 0; h < hiddenUnits; h++) {
          sum += hiddenLayer[h] * weightsHiddenToOutput[h][outIdx];
        }
        rawOutputs[outIdx] = sum;
      }

      const maxOutput = Math.max(...rawOutputs);
      const expOutputs = rawOutputs.map(val => Math.exp(val - maxOutput));
      const sumExpOutputs = expOutputs.reduce((acc, val) => acc + val, 0);
      const probabilities = expOutputs.map(val => val / sumExpOutputs);

      totalLoss -= Math.log(probabilities[targetId] + 1e-7);

      const outputErrors = [...probabilities];
      outputErrors[targetId] -= 1;

      const hiddenErrors = new Array(hiddenUnits).fill(0);
      for (let h = 0; h < hiddenUnits; h++) {
        let errorSum = 0;
        for (let outIdx = 0; outIdx < vocabSize; outIdx++) {
          errorSum += outputErrors[outIdx] * weightsHiddenToOutput[h][outIdx];
        }
        hiddenErrors[h] = hiddenLayer[h] > 0 ? errorSum : 0;
      }

      for (let h = 0; h < hiddenUnits; h++) {
        for (let outIdx = 0; outIdx < vocabSize; outIdx++) {
          weightsHiddenToOutput[h][outIdx] -= learningRate * outputErrors[outIdx] * hiddenLayer[h];
        }
      }

      for (let inIdx = 0; inIdx < inputVectorSize; inIdx++) {
        for (let h = 0; h < hiddenUnits; h++) {
          weightsInputToHidden[inIdx][h] -= learningRate * hiddenErrors[h] * xVector[inIdx];
        }
      }
    }

    if ((epoch + 1) % 20 === 0 || epoch === totalEpochs - 1) {
      const averageLoss = totalLoss / dataset.length;
      self.postMessage({
        type: 'progress',
        currentEpoch: epoch + 1,
        loss: averageLoss
      });
    }
  }

  self.postMessage({
    type: 'complete',
    model: {
      weightsInputToHidden,
      weightsHiddenToOutput,
      charToId,
      idToChar,
      sequenceLength,
      vocabSize,
      hiddenUnits
    }
  });
};
