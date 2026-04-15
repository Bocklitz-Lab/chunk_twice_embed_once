'use client';

import type { ClockPayload } from '@shared-types';
import { useMemo } from 'react';

import '../styles/clock.css';

type Props = {
  clock: ClockPayload;
};

function formatOffset(seconds: number): string {
  const sign = seconds >= 0 ? '+' : '-';
  const abs = Math.abs(seconds);
  const hours = String(Math.floor(abs / 3600)).padStart(2, '0');
  const minutes = String(Math.floor((abs % 3600) / 60)).padStart(2, '0');
  return `${sign}${hours}:${minutes}`;
}

export function ClockCard({ clock }: Props): JSX.Element {
  const offsetLabel = useMemo(() => formatOffset(clock.utcOffsetSeconds), [clock.utcOffsetSeconds]);

  return (
    <article className={`clock-card${clock.isLocal ? ' clock-card--local' : ''}`} data-testid="clock-item">
      <header className="clock-card__header">
        <h2>{clock.label}</h2>
        <span className="clock-card__offset" aria-label={`UTC offset ${offsetLabel}`}>
          UTC {offsetLabel}
        </span>
      </header>
      <p className="clock-card__time" aria-live={clock.isLocal ? 'polite' : 'off'}>
        {new Date(clock.localTimeISO).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
      </p>
      {clock.nextTransitionAt && (
        <p className="clock-card__transition">
          Next change: {new Date(clock.nextTransitionAt).toLocaleString()}
        </p>
      )}
    </article>
  );
}

export default ClockCard;
