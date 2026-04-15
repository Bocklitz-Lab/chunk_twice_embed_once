import { PrismaClient } from '@prisma/client';
import { Temporal } from '@js-temporal/polyfill';

const prisma = new PrismaClient();

async function main(): Promise<void> {
  const effectiveAt = Temporal.Now.instant();
  const snapshot = await prisma.timeSourceSnapshot.upsert({
    where: { checksum: 'bootstrap-000' },
    update: {},
    create: {
      effectiveAt: effectiveAt.toZonedDateTimeISO('UTC').toInstant().toString(),
      sourceVersion: 'bootstrap',
      checksum: 'bootstrap-000',
      metadata: { source: 'seed', notes: 'Initial bootstrap snapshot' },
      entries: {
        create: [
          {
            timezone: 'UTC',
            utcOffsetSeconds: 0,
            nextTransitionAt: null,
            nextOffsetSeconds: 0
          },
          {
            timezone: 'America/New_York',
            utcOffsetSeconds: -14400,
            nextTransitionAt: null,
            nextOffsetSeconds: -18000
          },
          {
            timezone: 'Asia/Tokyo',
            utcOffsetSeconds: 32400,
            nextTransitionAt: null,
            nextOffsetSeconds: 32400
          }
        ]
      }
    }
  });

  await prisma.clockDisplayView.upsert({
    where: { id: 'seed-view' },
    update: {},
    create: {
      id: 'seed-view',
      user: {
        create: {
          id: 'seed-user',
          theme: 'light',
          locale: 'en-US'
        }
      },
      snapshotId: snapshot.id,
      payload: {
        clocks: [
          { id: 'local', label: 'Local', timezone: 'UTC', utcOffsetSeconds: 0 },
          { id: 'nyc', label: 'New York', timezone: 'America/New_York', utcOffsetSeconds: -14400 },
          { id: 'tokyo', label: 'Tokyo', timezone: 'Asia/Tokyo', utcOffsetSeconds: 32400 }
        ]
      },
      referenceTimestamp: effectiveAt.toString(),
      accuracyBudgetMs: 60000
    }
  });

  console.info('Seed data inserted.');
}

main()
  .catch((error) => {
    console.error('Seed failure', error);
    process.exitCode = 1;
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
