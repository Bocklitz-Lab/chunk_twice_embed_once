import { Server, Socket } from 'socket.io';

import { getClocks } from '../services/clocksService';

export function registerClockTicker(io: Server): void {
  io.on('connection', (socket: Socket) => {
    console.info(`[socket] client connected ${socket.id}`);
    const includeParam = socket.handshake.query.include;
    const include = Array.isArray(includeParam)
      ? includeParam.flatMap((value) => value.split(',').map((segment) => segment.trim()))
      : typeof includeParam === 'string'
      ? includeParam.split(',').map((segment) => segment.trim())
      : [];

    const localTimezone = (socket.handshake.headers['x-local-timezone'] as string | undefined) ?? null;

    let active = true;

    const tick = async () => {
      try {
        const payload = await getClocks({ include, localTimezone });
        if (active) {
          socket.emit('clock.tick', payload);
        }
      } catch (error) {
        socket.emit('clock.error', { message: (error as Error).message });
      }
    };

    const interval = setInterval(tick, 1_000);
    void tick();

    socket.on('disconnect', () => {
      active = false;
      clearInterval(interval);
      console.info(`[socket] client disconnected ${socket.id}`);
    });
  });
}
