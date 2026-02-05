import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getEventInfo, getEventTeams, getEventMatches, getEventTeamStats, getTeamsYearStats } from '../../lib/api';
import { getEventScoutingData } from '../../lib/scouting';
import { TEAM_NUMBER } from '../../lib/supabase';
import TeamList from '../../components/scout4308/TeamList';
import MatchList from '../../components/scout4308/MatchList';

const EventDashboard = () => {
  const { eventKey } = useParams();
  const [activeTab, setActiveTab] = useState('teams');
  const [eventInfo, setEventInfo] = useState(null);
  const [teams, setTeams] = useState([]);
  const [matches, setMatches] = useState([]);
  const [teamStats, setTeamStats] = useState({});
  const [teamYearStats, setTeamYearStats] = useState({});
  const [scoutingData, setScoutingData] = useState({});
  const [loading, setLoading] = useState(true);
  const [loadingStats, setLoadingStats] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch all data in parallel
        const [info, teamsData, matchesData, statsData, scoutData] = await Promise.all([
          getEventInfo(eventKey),
          getEventTeams(eventKey),
          getEventMatches(eventKey),
          getEventTeamStats(eventKey),
          getEventScoutingData(eventKey),
        ]);

        setEventInfo(info);
        const sortedTeams = teamsData.sort((a, b) => a.team_number - b.team_number);
        setTeams(sortedTeams);
        setMatches(matchesData);
        
        // Convert stats array to object keyed by team number
        const statsMap = {};
        if (statsData) {
          statsData.forEach(stat => {
            statsMap[stat.team] = stat;
          });
        }
        setTeamStats(statsMap);

        // Convert scouting data array to object keyed by team number
        const scoutMap = {};
        if (scoutData) {
          scoutData.forEach(data => {
            scoutMap[data.team_number] = data;
          });
        }
        setScoutingData(scoutMap);

        // Fetch year stats in background (with fallback)
        setLoadingStats(true);
        const teamNumbers = sortedTeams.map(t => t.team_number);
        getTeamsYearStats(teamNumbers).then(yearStats => {
          setTeamYearStats(yearStats);
          setLoadingStats(false);
        });

      } catch (err) {
        setError('Failed to load event data.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [eventKey]);

  const updateScoutingData = (teamNumber, data) => {
    setScoutingData(prev => ({
      ...prev,
      [teamNumber]: data,
    }));
  };

  if (loading) {
    return (
      <main className="flex-grow flex items-center justify-center">
        <p className="text-sm text-ink/70 uppercase tracking-terminal">
          [ LOADING EVENT DATA... ]
        </p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="flex-grow flex flex-col items-center justify-center gap-4">
        <p className="text-sm text-red-600 uppercase tracking-terminal">[ {error} ]</p>
        <Link to="/scout4308" className="text-sm uppercase tracking-terminal text-ink hover:underline">
          ← BACK TO EVENTS
        </Link>
      </main>
    );
  }

  return (
    <main className="flex-grow flex flex-col px-6 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <Link to="/scout4308" className="text-xs uppercase tracking-terminal text-ink/50 hover:text-ink mb-2 inline-block">
          ← BACK TO EVENTS
        </Link>
        <h1 className="text-xl font-bold uppercase tracking-terminal text-ink">
          [ {eventInfo?.name || eventKey} ]
        </h1>
        <p className="text-xs text-ink/70 uppercase tracking-terminal mt-1">
          {eventInfo?.city}, {eventInfo?.state_prov} | {eventInfo?.start_date}
        </p>
        {loadingStats && (
          <p className="text-xs text-ink/50 uppercase tracking-terminal mt-2">
            [ LOADING STATBOTICS DATA... ]
          </p>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => setActiveTab('teams')}
          className={`px-6 py-3 text-sm uppercase tracking-terminal font-semibold border transition-none ${
            activeTab === 'teams'
              ? 'bg-highlight text-invert border-highlight'
              : 'border-structure hover:bg-highlight hover:text-invert hover:border-highlight'
          }`}
        >
          {activeTab === 'teams' ? '[*]' : '[ ]'} TEAMS ({teams.length})
        </button>
        <button
          onClick={() => setActiveTab('matches')}
          className={`px-6 py-3 text-sm uppercase tracking-terminal font-semibold border transition-none ${
            activeTab === 'matches'
              ? 'bg-highlight text-invert border-highlight'
              : 'border-structure hover:bg-highlight hover:text-invert hover:border-highlight'
          }`}
        >
          {activeTab === 'matches' ? '[*]' : '[ ]'} MATCHES ({matches.length})
        </button>
      </div>

      {/* Content */}
      {activeTab === 'teams' && (
        <TeamList 
          teams={teams} 
          teamStats={teamStats}
          teamYearStats={teamYearStats}
          scoutingData={scoutingData}
          eventKey={eventKey}
          onScoutingUpdate={updateScoutingData}
        />
      )}
      {activeTab === 'matches' && (
        <MatchList 
          matches={matches} 
          teamStats={teamStats}
          teamNumber={TEAM_NUMBER}
        />
      )}
    </main>
  );
};

export default EventDashboard;
