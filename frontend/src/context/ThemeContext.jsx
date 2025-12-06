import { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext(null);

export const ThemeProvider = ({ children }) => {
    const [isDarkMode, setIsDarkMode] = useState(() => {
        const saved = localStorage.getItem('theme');
        return saved ? saved === 'dark' : false;
    });

    useEffect(() => {
        localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
        document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
    }, [isDarkMode]);

    const toggleTheme = () => setIsDarkMode(!isDarkMode);

    const theme = {
        isDarkMode,
        toggleTheme,
        colors: isDarkMode ? {
            // Dark mode colors
            bg: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%)',
            cardBg: '#1e1e2e',
            cardBorder: '#2d2d44',
            text: '#e4e4e7',
            textMuted: '#a1a1aa',
            primary: '#818cf8',
            primaryGradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            inputBg: '#252536',
            inputBorder: '#3d3d5c',
            messageBg: '#2a2a3e',
            shadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)',
        } : {
            // Light mode colors
            bg: 'linear-gradient(135deg, #e0f2fe 0%, #e0e7ff 50%, #f3e8ff 100%)',
            cardBg: '#ffffff',
            cardBorder: '#e5e7eb',
            text: '#111827',
            textMuted: '#6b7280',
            primary: '#667eea',
            primaryGradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            inputBg: '#ffffff',
            inputBorder: '#d1d5db',
            messageBg: '#f3f4f6',
            shadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
        }
    };

    return (
        <ThemeContext.Provider value={theme}>
            {children}
        </ThemeContext.Provider>
    );
};

export const useTheme = () => {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within ThemeProvider');
    }
    return context;
};
