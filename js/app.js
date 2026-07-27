let trainedModel = null;
let activeWorker = null;

const trainBtn = document.getElementById('trainBtn');
const trainBtnText = document.getElementById('trainBtnText');
const trainIcon = document.getElementById('trainIcon');
const generateBtn = document.getElementById('generateBtn');
const statusText = document.getElementById('statusText');
const textInput = document.getElementById('textInput');
const seedInput = document.getElementById('seedInput');

// 生成文章の最大文字数制限
const MAX_CHAR_LIMIT = 100;

// 起動時に /data/gakusyuu1.json を取得して学習用テキストエリアへ読み込み
// directionは任意パラメータとして処理し、大量のinfoテキストをそのまま結合
window.addEventListener('DOMContentLoaded', () => {
  fetch('/data/gakusyuu1.json')
    .then(response => {
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json();
    })
    .then(data => {
      if (Array.isArray(data)) {
        // infoが存在する場合はそのまま抽出し、文字列配列として取得
        const textLines = data.map(item => {
          if (typeof item === 'string') return item;
          if (item && item.info) return item.info;
          return '';
        }).filter(Boolean);
        
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

// 重複文章・連続重複単語の自動削除フィルター
function removeDuplicatePhrases(tokens) {
  const result = [];
  for (let i = 0; i < tokens.length; i++) {
    const currentToken = tokens[i];
    // 同じ単語が連続した場合は追加せずスキップ
    if (result.length > 0 && result[result.length - 1] === currentToken) {
      continue;
    }
    
    // 2単語以上のループパターン検出（例: A B A B）
    if (result.length >= 2 && i < tokens.length - 1) {
      const prevTwo = result.slice(-2).join(' ');
      const nextTwo = [tokens[i], tokens[i + 1]].join(' ');
      if (prevTwo === nextTwo) {
        break; // ループ検出時点で生成終了
      }
    }
    
    result.push(currentToken);
  }
  return result;
}

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
  const generateLength = 30; // 内部生成ステップ数

  // 非同期で生成
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

      // 出力スコア順にインデックスをソート
      const sortedIndices = Array.from({ length: vocabSize }, (_, k) => k)
        .sort((a, b) => rawOutputs[b] - rawOutputs[a]);

      let bestCharId = sortedIndices[0];
      const lastToken = generatedTokens[generatedTokens.length - 1];

      // 直前の単語と同じ単語が出力されそうな場合、次の高スコア単語を選択して重複を防止
      if (idToChar[bestCharId] === lastToken && sortedIndices.length > 1) {
        bestCharId = sortedIndices[1];
      }

      const predictedChar = idToChar[bestCharId] || '';
      generatedTokens.push(predictedChar);
    }

    // 同じ単語・フレーズの繰り返しを削除
    const filteredTokens = removeDuplicatePhrases(generatedTokens);
    let generatedResult = filteredTokens.join(' ');

    // 文字数制限の適用 (MAX_CHAR_LIMIT 文字を超えた場合は切り捨て)
    if (generatedResult.length > MAX_CHAR_LIMIT) {
      generatedResult = generatedResult.substring(0, MAX_CHAR_LIMIT);
    }

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
