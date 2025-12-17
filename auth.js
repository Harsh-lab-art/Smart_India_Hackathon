// API Base URL
const API_URL = 'http://localhost:5000';

// Utility Functions
function showMessage(elementId, message, type) {
    const messageEl = document.getElementById(elementId);
    messageEl.textContent = message;
    messageEl.className = `message ${type}`;
    messageEl.style.display = 'block';
    
    setTimeout(() => {
        messageEl.style.display = 'none';
    }, 5000);
}

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// Sign Up
if (document.getElementById('signupForm')) {
    document.getElementById('signupForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const name = document.getElementById('signupName').value;
        const email = document.getElementById('signupEmail').value;
        const password = document.getElementById('signupPassword').value;
        const confirmPassword = document.getElementById('signupConfirmPassword').value;
        
        // Validation
        if (!validateEmail(email)) {
            showMessage('signupMessage', 'Please enter a valid email address', 'error');
            return;
        }
        
        if (password.length < 8) {
            showMessage('signupMessage', 'Password must be at least 8 characters', 'error');
            return;
        }
        
        if (password !== confirmPassword) {
            showMessage('signupMessage', 'Passwords do not match', 'error');
            return;
        }
        
        try {
            const response = await fetch(`${API_URL}/auth/signup`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name, email, password })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showMessage('signupMessage', 'Account created successfully! Redirecting...', 'success');
                setTimeout(() => {
                    window.location.href = 'login.html';
                }, 2000);
            } else {
                showMessage('signupMessage', data.error || 'Sign up failed', 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showMessage('signupMessage', 'Network error. Please try again.', 'error');
        }
    });
}

// Sign In
if (document.getElementById('loginForm')) {
    document.getElementById('loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        const rememberMe = document.getElementById('rememberMe').checked;
        
        if (!validateEmail(email)) {
            showMessage('loginMessage', 'Please enter a valid email address', 'error');
            return;
        }
        
        try {
            const response = await fetch(`${API_URL}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, password })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Store user data
                if (rememberMe) {
                    localStorage.setItem('user', JSON.stringify(data.user));
                    localStorage.setItem('token', data.token);
                } else {
                    sessionStorage.setItem('user', JSON.stringify(data.user));
                    sessionStorage.setItem('token', data.token);
                }
                
                showMessage('loginMessage', 'Login successful! Redirecting...', 'success');
                setTimeout(() => {
                    window.location.href = 'index.html';
                }, 1500);
            } else {
                showMessage('loginMessage', data.error || 'Login failed', 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showMessage('loginMessage', 'Network error. Please try again.', 'error');
        }
    });
}

// Forgot Password Modal
let currentEmail = '';
let currentOTP = '';

function showForgotPassword() {
    document.getElementById('forgotPasswordModal').classList.add('active');
    document.getElementById('forgotStep1').style.display = 'block';
    document.getElementById('forgotStep2').style.display = 'none';
    document.getElementById('forgotStep3').style.display = 'none';
}

function closeForgotPassword() {
    document.getElementById('forgotPasswordModal').classList.remove('active');
    document.getElementById('forgotMessage').style.display = 'none';
    currentEmail = '';
    currentOTP = '';
}

// Step 1: Send OTP
if (document.getElementById('forgotEmailForm')) {
    document.getElementById('forgotEmailForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const email = document.getElementById('forgotEmail').value;
        
        if (!validateEmail(email)) {
            showMessage('forgotMessage', 'Please enter a valid email address', 'error');
            return;
        }
        
        try {
            const response = await fetch(`${API_URL}/auth/forgot-password`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                currentEmail = email;
                document.getElementById('otpEmail').textContent = email;
                document.getElementById('forgotStep1').style.display = 'none';
                document.getElementById('forgotStep2').style.display = 'block';
                showMessage('forgotMessage', 'OTP sent to your email!', 'success');
            } else {
                showMessage('forgotMessage', data.error || 'Failed to send OTP', 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showMessage('forgotMessage', 'Network error. Please try again.', 'error');
        }
    });
}

// Step 2: Verify OTP
if (document.getElementById('verifyOtpForm')) {
    document.getElementById('verifyOtpForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const otp = document.getElementById('otpCode').value;
        
        if (otp.length !== 6) {
            showMessage('forgotMessage', 'Please enter a 6-digit OTP', 'error');
            return;
        }
        
        try {
            const response = await fetch(`${API_URL}/auth/verify-otp`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email: currentEmail, otp })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                currentOTP = otp;
                document.getElementById('forgotStep2').style.display = 'none';
                document.getElementById('forgotStep3').style.display = 'block';
                showMessage('forgotMessage', 'OTP verified! Enter new password', 'success');
            } else {
                showMessage('forgotMessage', data.error || 'Invalid OTP', 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showMessage('forgotMessage', 'Network error. Please try again.', 'error');
        }
    });
}

// Resend OTP
async function resendOTP() {
    try {
        const response = await fetch(`${API_URL}/auth/forgot-password`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email: currentEmail })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showMessage('forgotMessage', 'New OTP sent to your email!', 'success');
        } else {
            showMessage('forgotMessage', data.error || 'Failed to resend OTP', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage('forgotMessage', 'Network error. Please try again.', 'error');
    }
}

// Step 3: Reset Password
if (document.getElementById('resetPasswordForm')) {
    document.getElementById('resetPasswordForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const newPassword = document.getElementById('newPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        
        if (newPassword.length < 8) {
            showMessage('forgotMessage', 'Password must be at least 8 characters', 'error');
            return;
        }
        
        if (newPassword !== confirmPassword) {
            showMessage('forgotMessage', 'Passwords do not match', 'error');
            return;
        }
        
        try {
            const response = await fetch(`${API_URL}/auth/reset-password`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    email: currentEmail, 
                    otp: currentOTP, 
                    newPassword 
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showMessage('forgotMessage', 'Password reset successful! Redirecting...', 'success');
                setTimeout(() => {
                    closeForgotPassword();
                    window.location.href = 'login.html';
                }, 2000);
            } else {
                showMessage('forgotMessage', data.error || 'Failed to reset password', 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showMessage('forgotMessage', 'Network error. Please try again.', 'error');
        }
    });
}

// Close modal when clicking outside
if (document.getElementById('forgotPasswordModal')) {
    document.getElementById('forgotPasswordModal').addEventListener('click', (e) => {
        if (e.target.id === 'forgotPasswordModal') {
            closeForgotPassword();
        }
    });
}
