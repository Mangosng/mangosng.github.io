import { Link, useLocation } from 'react-router-dom';

const Header = () => {
  const location = useLocation();

  const getLinkClass = (path) => {
    const isActive = location.pathname === path;
    return isActive
      ? "bg-ink text-invert px-3 py-1 uppercase tracking-terminal font-semibold"
      : "text-ink uppercase tracking-terminal font-semibold hover:bg-ink hover:text-invert px-3 py-1 transition-none";
  };

  return (
    <header className="bg-canvas border-b border-dotted border-structure sticky top-0 z-50">
      <div className="container mx-auto flex justify-between items-center py-4 px-6">
        <span className="text-ink font-semibold uppercase tracking-terminal text-sm">
          [ MANAV.IO ]
        </span>
        <nav className="flex space-x-6 text-sm">
          <Link to="/" className={getLinkClass("/")}>[ HOME ]</Link>
          <Link to="/projects" className={getLinkClass("/projects")}>[ PROJECTS ]</Link>
          <Link to="/cv" className={getLinkClass("/cv")}>[ CV ]</Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
