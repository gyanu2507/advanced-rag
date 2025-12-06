import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Mail, Lock, Phone, Loader2, Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { authAPI } from '../services/api';
import styled from 'styled-components';

const Container = styled.div`
  min-height: 100vh;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #e0f2fe 0%, #e0e7ff 50%, #f3e8ff 100%);
  padding: 1rem;
  margin: 0;
  box-sizing: border-box;
`;

const Card = styled.div`
  width: 100%;
  max-width: 28rem;
  background: white;
  border-radius: 1rem;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  padding: 2rem;
  margin: 0 auto;
`;

const Title = styled.h1`
  font-size: 1.875rem;
  font-weight: 700;
  color: #111827;
  margin-bottom: 0.5rem;
  text-align: center;
`;

const Subtitle = styled.p`
  color: #4b5563;
  text-align: center;
  margin-bottom: 2rem;
`;

const ToggleContainer = styled.div`
  display: flex;
  background: #f3f4f6;
  border-radius: 0.5rem;
  padding: 0.25rem;
  margin-bottom: 1.5rem;
`;

const ToggleButton = styled.button`
  flex: 1;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-weight: 600;
  transition: all 0.2s;
  border: none;
  cursor: pointer;
  background: ${props => props.active ? 'white' : 'transparent'};
  color: ${props => props.active ? '#111827' : '#4b5563'};
  box-shadow: ${props => props.active ? '0 1px 2px 0 rgba(0, 0, 0, 0.05)' : 'none'};
  
  &:hover {
    color: #111827;
  }
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const Label = styled.label`
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  margin-bottom: 0.375rem;
`;

const InputWrapper = styled.div`
  position: relative;
  width: 100%;
`;

const Input = styled.input`
  width: 100%;
  padding: 0.625rem 0.75rem;
  padding-left: ${props => props.hasIcon ? '2.5rem' : '0.75rem'};
  padding-right: ${props => props.hasPassword ? '2.5rem' : '0.75rem'};
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: #111827;
  outline: none;
  transition: all 0.2s;
  
  &:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
  
  &::placeholder {
    color: #9ca3af;
  }
`;

const IconWrapper = styled.div`
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  color: #9ca3af;
  pointer-events: none;
`;

const PasswordToggle = styled.button`
  position: absolute;
  right: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  color: #9ca3af;
  cursor: pointer;
  padding: 0;
  
  &:hover {
    color: #4b5563;
  }
`;

const CheckboxContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const CheckboxLabel = styled.label`
  display: flex;
  align-items: center;
  font-size: 0.875rem;
  color: #4b5563;
  cursor: pointer;
`;

const Checkbox = styled.input`
  margin-right: 0.5rem;
  border-radius: 0.25rem;
  border: 1px solid #d1d5db;
  accent-color: #3b82f6;
`;

const Link = styled.a`
  font-size: 0.875rem;
  color: #2563eb;
  font-weight: 500;
  text-decoration: none;
  
  &:hover {
    color: #1d4ed8;
    text-decoration: underline;
  }
`;

const Button = styled.button`
  width: 100%;
  padding: 0.625rem 1rem;
  background: ${props => props.variant === 'primary' ? '#111827' : props.variant === 'secondary' ? 'transparent' : '#f3f4f6'};
  color: ${props => props.variant === 'primary' ? 'white' : props.variant === 'secondary' ? '#374151' : '#374151'};
  border: ${props => props.variant === 'secondary' ? '2px solid #d1d5db' : 'none'};
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  
  &:hover {
    background: ${props => props.variant === 'primary' ? '#1f2937' : props.variant === 'secondary' ? '#f9fafb' : '#e5e7eb'};
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px ${props => props.variant === 'primary' ? 'rgba(17, 24, 39, 0.2)' : 'rgba(209, 213, 219, 0.5)'};
  }
`;

const Divider = styled.div`
  position: relative;
  margin: 1.5rem 0;
  
  &::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 1px;
    background: #d1d5db;
  }
`;

const DividerText = styled.div`
  position: relative;
  display: flex;
  justify-content: center;
  font-size: 0.875rem;
  
  span {
    background: white;
    padding: 0 0.5rem;
    color: #6b7280;
  }
`;

const ErrorMessage = styled.div`
  padding: 0.75rem;
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #991b1b;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  margin-bottom: 1rem;
`;

const OTPInput = styled.input`
  width: 100%;
  padding: 0.625rem 1rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 1.25rem;
  letter-spacing: 0.5em;
  text-align: center;
  font-weight: 700;
  color: #111827;
  outline: none;
  transition: all 0.2s;
  
  &:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
`;

const InfoBox = styled.div`
  padding: 0.75rem;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  color: #1e40af;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
`;

const LoginPage = () => {
  const { login, guestLogin, user, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [phone, setPhone] = useState('');
  const [otp, setOtp] = useState('');
  const [otpSent, setOtpSent] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [googleConfig, setGoogleConfig] = useState(null);
  const [guestLoginTriggered, setGuestLoginTriggered] = useState(false);

  // Navigate to dashboard when guest login completes
  useEffect(() => {
    if (guestLoginTriggered && isAuthenticated && user && user.user_id) {
      console.log('Guest login successful, navigating to dashboard', { user, isAuthenticated });
      navigate('/');
      setGuestLoginTriggered(false);
    }
  }, [guestLoginTriggered, isAuthenticated, user, navigate]);

  useEffect(() => {
    // Load Google config
    authAPI.getGoogleConfig().then((config) => {
      console.log('Google Config:', config);
      setGoogleConfig(config);
    }).catch((err) => {
      console.error('Failed to load Google config:', err);
      setGoogleConfig(null);
    });

    // Wait for Google script to load
    const initGoogleSignIn = () => {
      if (window.google && googleConfig?.client_id && googleConfig?.enabled) {
        window.google.accounts.id.initialize({
          client_id: googleConfig.client_id,
          callback: async (response) => {
            try {
              const result = await authAPI.googleAuth(response.credential);
              if (result.status === 'success') {
                login(result.token, result.user);
                window.location.href = '/';
              }
            } catch (err) {
              setError('Google authentication failed');
            }
          },
        });
        window.google.accounts.id.renderButton(
          document.getElementById('google-signin-button'),
          {
            type: 'standard',
            size: 'large',
            theme: 'outline',
            text: 'sign_in_with',
            shape: 'rectangular',
            logo_alignment: 'left',
          }
        );
      }
    };

    // Check if Google script is already loaded
    if (window.google) {
      initGoogleSignIn();
    } else {
      // Wait for script to load
      const checkGoogle = setInterval(() => {
        if (window.google) {
          clearInterval(checkGoogle);
          initGoogleSignIn();
        }
      }, 100);

      // Cleanup after 10 seconds
      setTimeout(() => clearInterval(checkGoogle), 10000);
    }
  }, [googleConfig]);

  const handleEmailAuth = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      let result;
      if (mode === 'signup') {
        result = await authAPI.signup(email, password);
      } else {
        result = await authAPI.login(email, password);
      }

      if (result.status === 'success') {
        login(result.token, result.user);
        window.location.href = '/';
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const handlePhoneOTP = async () => {
    if (!otpSent) {
      setLoading(true);
      try {
        const result = await authAPI.sendOTP(phone);
        if (result.status === 'success') {
          setOtpSent(true);
        } else {
          setError('Failed to send OTP');
        }
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to send OTP');
      } finally {
        setLoading(false);
      }
    } else {
      setLoading(true);
      try {
        const result = await authAPI.verifyOTP(phone, otp);
        if (result.status === 'success') {
          login(result.token, result.user);
          window.location.href = '/';
        } else {
          setError('Invalid OTP code');
        }
      } catch (err) {
        setError(err.response?.data?.detail || 'Verification failed');
      } finally {
        setLoading(false);
      }
    }
  };

  // Re-initialize Google Sign-In when config changes
  useEffect(() => {
    if (googleConfig?.client_id && googleConfig?.enabled && window.google) {
      const buttonElement = document.getElementById('google-signin-button');
      if (buttonElement) {
        // Clear any existing button
        buttonElement.innerHTML = '';
        
        window.google.accounts.id.initialize({
          client_id: googleConfig.client_id,
          callback: async (response) => {
            try {
              const result = await authAPI.googleAuth(response.credential);
              if (result.status === 'success') {
                login(result.token, result.user);
                window.location.href = '/';
              }
            } catch (err) {
              setError('Google authentication failed: ' + (err.response?.data?.detail || err.message));
            }
          },
        });
        
        window.google.accounts.id.renderButton(
          buttonElement,
          {
            type: 'standard',
            size: 'large',
            theme: 'outline',
            text: 'sign_in_with',
            shape: 'rectangular',
            logo_alignment: 'left',
          }
        );
      }
    }
  }, [googleConfig, login]);

  return (
    <Container>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        style={{ width: '100%', maxWidth: '28rem', margin: '0 auto' }}
      >
        <Card>
          <Title>Welcome Back</Title>
          <Subtitle>
            {mode === 'login' ? 'Sign in to your account' : 'Create a new account'}
          </Subtitle>

          <ToggleContainer>
            <ToggleButton
              active={mode === 'login'}
              onClick={() => {
                setMode('login');
                setError('');
              }}
            >
              Sign In
            </ToggleButton>
            <ToggleButton
              active={mode === 'signup'}
              onClick={() => {
                setMode('signup');
                setError('');
              }}
            >
              Sign Up
            </ToggleButton>
          </ToggleContainer>

          {error && <ErrorMessage>{error}</ErrorMessage>}

          <Form onSubmit={handleEmailAuth}>
            <div>
              <Label>Email</Label>
              <InputWrapper>
                <IconWrapper>
                  <Mail size={18} />
                </IconWrapper>
                <Input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email"
                  hasIcon
                  required
                />
              </InputWrapper>
            </div>

            <div>
              <Label>Password</Label>
              <InputWrapper>
                <IconWrapper>
                  <Lock size={18} />
                </IconWrapper>
                <Input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  hasIcon
                  hasPassword
                  required
                />
                <PasswordToggle
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </PasswordToggle>
              </InputWrapper>
            </div>

            {mode === 'login' && (
              <CheckboxContainer>
                <CheckboxLabel>
                  <Checkbox type="checkbox" />
                  Remember me
                </CheckboxLabel>
                <Link href="#">Forgot password?</Link>
              </CheckboxContainer>
            )}

            <Button type="submit" variant="primary" disabled={loading}>
              {loading ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <span>Sign In</span>
              )}
            </Button>
          </Form>

          <Divider>
            <DividerText>
              <span>Or continue with</span>
            </DividerText>
          </Divider>

          {googleConfig?.client_id && googleConfig?.enabled ? (
            <div 
              id="google-signin-button" 
              style={{ 
                marginBottom: '1.5rem', 
                display: 'flex', 
                justifyContent: 'center',
                minHeight: '40px'
              }}
            />
          ) : (
            <div style={{ 
              marginBottom: '1.5rem', 
              padding: '0.75rem', 
              background: '#f3f4f6', 
              borderRadius: '0.5rem', 
              textAlign: 'center', 
              color: '#6b7280', 
              fontSize: '0.875rem' 
            }}>
              {googleConfig === null ? 'Loading...' : 'Google Sign-In not configured'}
            </div>
          )}

          <Button
            variant="secondary"
            onClick={() => {
              guestLogin();
              setGuestLoginTriggered(true);
            }}
            style={{ marginBottom: '1.5rem' }}
          >
            Continue as Guest
          </Button>

          <Divider>
            <DividerText>
              <span>Or</span>
            </DividerText>
          </Divider>

          <div>
            <Label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Phone size={16} />
              <span>Sign in with Phone</span>
            </Label>
            {!otpSent ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <InputWrapper>
                  <IconWrapper>
                    <Phone size={18} />
                  </IconWrapper>
                  <Input
                    type="tel"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    placeholder="+1234567890"
                    hasIcon
                  />
                </InputWrapper>
                <Button
                  type="button"
                  variant="secondary"
                  onClick={handlePhoneOTP}
                  disabled={loading || !phone}
                >
                  {loading ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      <span>Sending...</span>
                    </>
                  ) : (
                    <span>Send Code</span>
                  )}
                </Button>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <InfoBox>Code sent to {phone}</InfoBox>
                <OTPInput
                  type="text"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                  placeholder="000000"
                  maxLength={6}
                />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  <Button
                    type="button"
                    variant="primary"
                    onClick={handlePhoneOTP}
                    disabled={loading || otp.length !== 6}
                  >
                    Verify
                  </Button>
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={() => {
                      setOtpSent(false);
                      setOtp('');
                    }}
                  >
                    Resend
                  </Button>
                </div>
              </div>
            )}
          </div>
        </Card>
      </motion.div>
    </Container>
  );
};

export default LoginPage;
