function togglePassword() {
  const input = document.getElementById('password');
  const icon = document.getElementById('eye-icon');
  if (input.type === 'password') {
    input.type = 'text';
    icon.innerHTML = '<path d="M3 3l14 14M10.5 10.7A2.5 2.5 0 0113.3 13M6.5 6.6C4.6 7.8 3 10 3 10s3.5 6 7 6c1.5 0 2.9-.6 4-1.5" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/><path d="M10 4C5.5 4 2 10 2 10s1.2 2 3.2 3.8" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/><circle cx="10" cy="10" r="2.5" stroke="#94a3b8" stroke-width="1.5"/>';
  } else {
    input.type = 'password';
    icon.innerHTML = '<path d="M10 4C5.5 4 2 10 2 10s3.5 6 8 6 8-6 8-6-3.5-6-8-6z" stroke="#94a3b8" stroke-width="1.5"/><circle cx="10" cy="10" r="2.5" stroke="#94a3b8" stroke-width="1.5"/>';
  }
}

// Input focus animation
document.querySelectorAll('.input-wrapper input').forEach(input => {
  input.addEventListener('focus', () => {
    input.closest('.field-group').classList.add('focused');
  });
  input.addEventListener('blur', () => {
    input.closest('.field-group').classList.remove('focused');
  });
});

// Subtle form submit loading state
const form = document.querySelector('.login-form');
if (form) {
  form.addEventListener('submit', () => {
    const btn = form.querySelector('.btn-primary');
    if (btn) {
      btn.innerHTML = '<svg viewBox="0 0 20 20" fill="none" style="animation:spin 1s linear infinite;"><circle cx="10" cy="10" r="7" stroke="rgba(255,255,255,0.3)" stroke-width="2"/><path d="M10 3a7 7 0 017 7" stroke="white" stroke-width="2" stroke-linecap="round"/></svg> Authenticating…';
      btn.disabled = true;
    }
  });
}

const style = document.createElement('style');
style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
document.head.appendChild(style);
