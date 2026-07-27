// サイドバーおよびメッセージ領域のUI管理

const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const menuToggleBtn = document.getElementById('menuToggleBtn');
const closeSidebarBtn = document.getElementById('closeSidebarBtn');
const chatHistory = document.getElementById('chatHistory');

function openSidebar() {
  sidebar.classList.add('open');
  sidebarOverlay.classList.add('active');
}

function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('active');
}

menuToggleBtn.addEventListener('click', openSidebar);
closeSidebarBtn.addEventListener('click', closeSidebar);
sidebarOverlay.addEventListener('click', closeSidebar);

// メッセージ描画ヘルパー
function appendMessage(role, initialHtml = '') {
  const row = document.createElement('div');
  row.className = `message-row ${role}`;
  
  const avatar = document.createElement('div');
  avatar.className = `avatar ${role}`;
  
  const icon = document.createElement('span');
  icon.className = 'material-symbols-outlined';
  icon.style.fontSize = '1.2rem';
  icon.innerText = role === 'user' ? 'person' : 'smart_toy';
  avatar.appendChild(icon);

  const content = document.createElement('div');
  content.className = 'message-content';
  content.innerHTML = initialHtml;

  row.appendChild(avatar);
  row.appendChild(content);
  chatHistory.appendChild(row);

  chatHistory.scrollTop = chatHistory.scrollHeight;
  return content;
}
