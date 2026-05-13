import React, { useState, useEffect } from 'react';
import { getIncidents } from '../api';
import IncidentCard from './IncidentCard';

const IncidentFeed = () => {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchIncidents = async () => {
    try {
      const response = await getIncidents();
      if (response && response.status === 'ok') {
        // Sort by updated_at descending
        const sorted = [...response.data].sort((a, b) => 
          new Date(b.updated_at) - new Date(a.updated_at)
        );
        setIncidents(sorted);
      }
      setLoading(false);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch incidents:', err);
      setError('Connection lost to incident stream');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchIncidents();
    const interval = setInterval(fetchIncidents, 7000); // Poll every 7 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading && incidents.length === 0) {
    return (
      <div className="bg-[#0f172a]/40 border border-slate-800 rounded-2xl p-8 text-center">
        <div className="inline-block w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-slate-400 text-sm">Streaming live incidents...</p>
      </div>
    );
  }

  if (error && incidents.length === 0) {
    return (
      <div className="bg-red-500/5 border border-red-500/20 rounded-2xl p-8 text-center">
        <div className="text-red-400 text-2xl mb-2">⚠️</div>
        <p className="text-red-300 text-sm font-medium">{error}</p>
        <button 
          onClick={fetchIncidents}
          className="mt-4 px-4 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg text-xs transition-all border border-red-500/30"
        >
          Reconnect
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-2 px-1">
        <div className="flex items-center gap-2">
          <h2 className="text-xl font-bold text-slate-100 tracking-tight">Active Incidents</h2>
          <span className="px-2 py-0.5 bg-indigo-500/20 text-indigo-400 text-[10px] font-bold rounded-full border border-indigo-500/30 uppercase tracking-widest">Live</span>
        </div>
        <div className="text-[11px] text-slate-500">
          Showing {incidents.length} recent events
        </div>
      </div>

      {incidents.length === 0 ? (
        <div className="bg-[#1e293b]/30 border border-dashed border-slate-700 rounded-2xl p-12 text-center">
          <div className="text-4xl mb-4 opacity-20">🛡️</div>
          <h3 className="text-slate-300 font-medium mb-1">No Active Incidents</h3>
          <p className="text-slate-500 text-xs">Cluster is stable. Automated observers are monitoring.</p>
        </div>
      ) : (
        <div className="max-h-[800px] overflow-y-auto pr-2 custom-scrollbar">
          {incidents.map(incident => (
            <IncidentCard key={incident.id} incident={incident} />
          ))}
        </div>
      )}
    </div>
  );
};

export default IncidentFeed;
