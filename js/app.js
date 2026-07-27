let trainedModel = null;
let activeWorker = null;

const trainBtn = document.getElementById('trainBtn');
const trainBtnText = document.getElementById('trainBtnText');
const trainIcon = document.getElementById('trainIcon');
const generateBtn = document.getElementById('generateBtn');
const statusText = document.getElementById('statusText');
const textInput = document.getElementById('textInput');
const seedInput = document.getElementById('seedInput');

// 起動時に /data/gakusyuu1.json を取得して画面に初期ロード
window.addEventListener('DOMContentLoaded', () => {
  fetch('/data/gakusyuu1.json')
    .then(response => {
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json();
    })
    .then(data => {
      if (Array.isArray(data)) {
        const textLines = data.map(item => item.info).filter(Boolean);
        textInput.value = textLines.join('\n');
      }
    })
    .catch(error => {
      console.error('JSONデータの読み込みに失敗しました:', error);
    });
});

// 1. 学習処理
trainBtn.addEventListener('click', () => {
  const textData = textInput.value.replace(/\r?\n/g, ' ');
  if (textData.length < 5) {
    alert('学習用のテキストをもう少し長く入力してください。');
    return;
  }

  trainBtn.disabled = true;
  generateBtn.disabled = true;
  seedInput.disabled = true;
  trainIcon.innerText = 'sync';
  trainIcon.style.animation = 'spin 1s linear infinite';
  trainBtnText.innerText = '学習中...';

  statusText.innerText = '⚙️ バックグラウンド学習中...';

  if (activeWorker) activeWorker.terminate();
  activeWorker = new Worker('js/worker.js');

  activeWorker.onmessage = function(event) {
    const data = event.data;

    if (data.type === 'progress') {
      statusText.innerText = `🔥 学習進捗:\n[Epoch: ${data.currentEpoch}/300]\nLoss (誤差): ${data.loss.toFixed(4)}`;
    } else if (data.type === 'complete') {
      trainedModel = data.model;
      statusText.innerText = '✅ 学習完了！\nいつでもチャット可能です。';
      
      trainBtn.disabled = false;
      generateBtn.disabled = false;
      seedInput.disabled = false;

      trainIcon.innerText = 'auto_awesome';
      trainIcon.style.animation = 'none';
      trainBtnText.innerText = '再学習する';

      // スマホの場合は自動でサイドバーを閉じる
      closeSidebar();
    } else if (data.type === 'error') {
      alert(data.message);
      trainBtn.disabled = false;
      trainIcon.innerText = 'auto_awesome';
      trainIcon.style.animation = 'none';
      trainBtnText.innerText = '学習を開始する';
      statusText.innerText = '❌ エラーが発生しました';
    }
  };

  activeWorker.postMessage({
    textData: textData,
    totalEpochs: 300
  });
});

// 2. 推論・対話処理
function handleGenerate() {
  const prompt = seedInput.value.trim();
  if (!prompt || !trainedModel) return;

  // ユーザー発言を出力
  appendMessage('user', prompt);
  seedInput.value = '';

  // 思考中アニメーション表示
  const aiMessageContainer = appendMessage('assistant', `
    <div class="loading-dots">
      <span></span><span></span><span></span>
    </div>
  `);

  const {
    weightsInputToHidden,
    weightsHiddenToOutput,
    charToId,
    idToChar,
    sequenceLength,
    vocabSize,
    hiddenUnits
  } = trainedModel;

  let currentPromptTokens = prompt.split(/\s+/).filter(Boolean);
  let generatedTokens = [...currentPromptTokens];
  const generateLength = 15;

  // 非同期で生成（思考中表示を一瞬見せるため）
  setTimeout(() => {
    for (let step = 0; step < generateLength; step++) {
      let inputSubTokens = generatedTokens.slice(-sequenceLength);
      
      const fallbackKey = Object.keys(charToId)[0];
      while (inputSubTokens.length < sequenceLength) {
        inputSubTokens.unshift(fallbackKey);
      }

      const inputVectorSize = sequenceLength * vocabSize;
      const xVector = new Array(inputVectorSize).fill(0);
      
      inputSubTokens.forEach((char, pos) => {
        const id = charToId[char];
        if (id !== undefined) {
          xVector[pos * vocabSize + id] = 1;
        }
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

      let bestCharId = 0;
      let highestScore = -Infinity;
      for (let outIdx = 0; outIdx < vocabSize; outIdx++) {
        if (rawOutputs[outIdx] > highestScore) {
          highestScore = rawOutputs[outIdx];
          bestCharId = outIdx;
        }
      }

      const predictedChar = idToChar[bestCharId] || '';
      generatedTokens.push(predictedChar);
    }

    const generatedResult = generatedTokens.join(' ');

    // タイピングアニメーション開始
    aiMessageContainer.innerHTML = '';
    aiMessageContainer.classList.add('typing-cursor');

    let charIndex = 0;
    function typeEffect() {
      if (charIndex < generatedResult.length) {
        aiMessageContainer.innerText += generatedResult.charAt(charIndex);
        charIndex++;
        chatHistory.scrollTop = chatHistory.scrollHeight;
        setTimeout(typeEffect, 25);
      } else {
        // タイピング完了でカーソル消去
        aiMessageContainer.classList.remove('typing-cursor');
      }
    }
    typeEffect();
  }, 400);
}

generateBtn.addEventListener('click', handleGenerate);
seedInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    handleGenerate();
  }
});
