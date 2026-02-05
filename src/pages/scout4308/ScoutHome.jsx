import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getTeamEvents } from '../../lib/api';

const ScoutHome = () => {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [year, setYear] = useState(2026);

  useEffect(() => {
    const fetchEvents = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getTeamEvents(year);
        // Sort by start date
        const sorted = data.sort((a, b) => 
          new Date(a.start_date) - new Date(b.start_date)
        );
        setEvents(sorted);
      } catch (err) {
        setError('Failed to load events. Please try again.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchEvents();
  }, [year]);

  return (
    <main className="flex-grow flex flex-col items-center px-6 py-16">
      <h1 className="text-2xl font-bold uppercase tracking-terminal text-ink mb-4">
        [ SCOUT4308 ]
      </h1>
      <p className="text-sm text-ink/70 mb-8 text-center max-w-xl">
        FRC Team 4308 Scouting System
      </p>

      {/* Year selector */}
      <div className="flex items-center gap-4 mb-8">
        <span className="text-sm uppercase tracking-terminal text-ink">SEASON:</span>
        <select 
          value={year} 
          onChange={(e) => setYear(Number(e.target.value))}
          className="bg-canvas border border-structure px-4 py-2 text-ink font-mono text-sm uppercase"
        >
          <option value={2026}>2026</option>
          <option value={2025}>2025</option>
          <option value={2024}>2024</option>
          <option value={2023}>2023</option>
        </select>
      </div>

      <h2 className="text-sm font-bold uppercase tracking-terminal text-ink mb-6">
        [ SELECT COMPETITION ]
      </h2>

      {loading && (
        <p className="text-sm text-ink/70 uppercase tracking-terminal">
          [ LOADING EVENTS... ]
        </p>
      )}

      {error && (
        <p className="text-sm text-red-600 uppercase tracking-terminal">
          [ ERROR: {error} ]
        </p>
      )}

      {!loading && !error && events.length === 0 && (
        <p className="text-sm text-ink/70 uppercase tracking-terminal">
          [ NO EVENTS FOUND FOR {year} ]
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
        {events.map((event) => (
          <Link
            key={event.key}
            to={`/scout4308/event/${event.key}`}
            className="border border-structure p-6 hover:bg-highlight hover:text-invert hover:border-highlight transition-none"
          >
            <div className="text-xs uppercase tracking-terminal text-ink/50 mb-2">
              {event.start_date} → {event.end_date}
            </div>
            <h3 className="text-base font-bold uppercase tracking-terminal mb-1">
              {event.name}
            </h3>
            <div className="text-xs uppercase tracking-terminal text-ink/70">
              {event.city}, {event.state_prov}
            </div>
            <div className="text-xs uppercase tracking-terminal text-ink/50 mt-2">
              [ {event.key} ]
            </div>
          </Link>
        ))}
      </div>

      <Link 
        to="/projects" 
        className="mt-12 text-sm uppercase tracking-terminal text-ink/70 hover:text-ink"
      >
        ← BACK TO PROJECTS
      </Link>
    </main>
  );
};

export default ScoutHome;
