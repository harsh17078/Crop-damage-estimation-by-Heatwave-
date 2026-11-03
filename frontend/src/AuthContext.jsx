import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

const STORAGE_KEY = 'wheat_ai_auth_user';

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Restore session from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setUser(JSON.parse(stored));
      }
    } catch (e) {
      localStorage.removeItem(STORAGE_KEY);
    } finally {
      setLoading(false);
    }
  }, []);

  const login = (email, password) => {
    return new Promise((resolve, reject) => {
      // Simulate async auth — in production, replace with API call
      setTimeout(() => {
        // Demo credentials always work
        const isDemoCredentials =
          email === 'demo@wheatai.com' && password === 'demo1234';

        // Also allow any registered user from localStorage registry
        const registryRaw = localStorage.getItem('wheat_ai_users');
        const registry = registryRaw ? JSON.parse(registryRaw) : {};
        const registeredUser = registry[email.toLowerCase()];
        const isRegistered =
          registeredUser && registeredUser.password === password;

        if (isDemoCredentials) {
          const userData = {
            id: 'demo',
            name: 'Demo User',
            email: 'demo@wheatai.com',
            role: 'Enterprise Client',
            avatar: 'DU',
          };
          setUser(userData);
          localStorage.setItem(STORAGE_KEY, JSON.stringify(userData));
          resolve(userData);
        } else if (isRegistered) {
          const userData = {
            id: registeredUser.id,
            name: registeredUser.name,
            email: email.toLowerCase(),
            role: registeredUser.role || 'Client',
            avatar: registeredUser.name
              .split(' ')
              .map((n) => n[0])
              .join('')
              .toUpperCase()
              .slice(0, 2),
          };
          setUser(userData);
          localStorage.setItem(STORAGE_KEY, JSON.stringify(userData));
          resolve(userData);
        } else {
          reject(new Error('Invalid email or password.'));
        }
      }, 800);
    });
  };

  const register = (name, email, password) => {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (!name || !email || !password) {
          reject(new Error('All fields are required.'));
          return;
        }
        if (password.length < 6) {
          reject(new Error('Password must be at least 6 characters.'));
          return;
        }

        const registryRaw = localStorage.getItem('wheat_ai_users');
        const registry = registryRaw ? JSON.parse(registryRaw) : {};

        if (registry[email.toLowerCase()]) {
          reject(new Error('An account with this email already exists.'));
          return;
        }

        const newUser = {
          id: `user_${Date.now()}`,
          name,
          email: email.toLowerCase(),
          password,
          role: 'Client',
        };
        registry[email.toLowerCase()] = newUser;
        localStorage.setItem('wheat_ai_users', JSON.stringify(registry));

        const userData = {
          id: newUser.id,
          name: newUser.name,
          email: newUser.email,
          role: newUser.role,
          avatar: name
            .split(' ')
            .map((n) => n[0])
            .join('')
            .toUpperCase()
            .slice(0, 2),
        };
        setUser(userData);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(userData));
        resolve(userData);
      }, 800);
    });
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem(STORAGE_KEY);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
