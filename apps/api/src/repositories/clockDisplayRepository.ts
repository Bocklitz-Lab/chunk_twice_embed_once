import { prisma } from '../lib/prisma';

export async function getLatestSnapshot() {
  return prisma.timeSourceSnapshot.findFirst({
    orderBy: { effectiveAt: 'desc' },
    include: { entries: true }
  });
}

export async function getPinnedLocations(userId: string) {
  return prisma.pinnedLocation.findMany({
    where: { userId },
    orderBy: { orderIndex: 'asc' }
  });
}

export async function getUserProfile(userId: string) {
  return prisma.userProfile.findUnique({ where: { id: userId } });
}

export async function ensureUserProfile(userId: string) {
  const existing = await getUserProfile(userId);
  if (existing) return existing;
  return prisma.userProfile.create({ data: { id: userId } });
}
