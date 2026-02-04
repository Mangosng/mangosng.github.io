import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-canvas border-t border-dotted border-structure py-10 mt-20 text-center">
      <div className="flex justify-center space-x-6 mb-6 text-sm font-semibold uppercase tracking-terminal">
        <a href="https://instagram.com/manav.2009" target="_blank" rel="noopener noreferrer" className="text-ink hover:bg-ink hover:text-invert px-2 py-1 transition-none">
          [ INSTAGRAM ]
        </a>
        <a href="https://github.com/ManavSanghrajka" target="_blank" rel="noopener noreferrer" className="text-ink hover:bg-ink hover:text-invert px-2 py-1 transition-none">
          [ GITHUB ]
        </a>
        <a href="https://open.spotify.com/user/316ntxr3gfsimuv6cwvsg3dabxn4" target="_blank" rel="noopener noreferrer" className="text-ink hover:bg-ink hover:text-invert px-2 py-1 transition-none">
          [ SPOTIFY ]
        </a>
        <a href="https://linkedin.com/in/manav-sanghrajka-654a73363/" target="_blank" rel="noopener noreferrer" className="text-ink hover:bg-ink hover:text-invert px-2 py-1 transition-none">
          [ LINKEDIN ]
        </a>
        <a href="mailto:manavsanghrajka@gmail.com" className="text-ink hover:bg-ink hover:text-invert px-2 py-1 transition-none">
          [ EMAIL ]
        </a>
      </div>

      <p className="text-ink/50 text-xs tracking-terminal uppercase">
        -------- © 2025 MANAV | BUILT WITH REACT + VITE --------
      </p>
    </footer>
  );
};

export default Footer;
