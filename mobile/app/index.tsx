import { Feather, Ionicons, MaterialCommunityIcons, SimpleLineIcons } from '@expo/vector-icons';
import Constants from 'expo-constants';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import * as MediaLibrary from 'expo-media-library';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Animated,
  Image,
  Linking,
  Modal,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

type SyncStatus =
  | 'idle'
  | 'requesting-permission'
  | 'reading-camera-roll'
  | 'uploading'
  | 'analyzing_faces'
  | 'awaiting_face_pick'
  | 'extracting_clothing'
  | 'done'
  | 'failed';

type JobStatus = Exclude<SyncStatus, 'idle' | 'requesting-permission' | 'reading-camera-roll'>;
type MainTab = 'home' | 'search' | 'saved' | 'profile';
type SavedTab = 'wishlists' | 'items' | 'brands';
type HomeFeedTab = 'explore' | 'forYou' | 'trending';

interface PhotoOut {
  id: string;
  url: string;
  width: number | null;
  height: number | null;
}

interface FaceClusterOut {
  id: string;
  rep_photo_id: string;
  rep_bbox: { left: number; top: number; width: number; height: number };
  rep_aspect_ratio: number;
  member_count: number;
  source_url: string;
}

interface ClothingItemOut {
  id: string;
  photo_id: string;
  category: string;
  description: string;
  colors: string[];
  pattern: string;
  style: string;
  brand_visible: string | null;
  visibility: string;
  confidence: number;
  bounding_box: { x: number; y: number; w: number; h: number };
  crop_url: string | null;
  tier: string;
  exact_matches: Record<string, unknown>[];
  similar_products: Record<string, unknown>[];
  best_match: {
    title: string;
    source: string;
    price: string | null;
    link: string;
    thumbnail: string;
    confidence: number;
    reasoning: string;
    source_tier: 'exact' | 'similar';
  } | null;
  best_match_confidence: number;
}

interface JobDetail {
  id: string;
  status: JobStatus;
  photo_count: number;
  error: string | null;
  created_at: string;
  updated_at: string;
  selected_cluster_id: string | null;
  photos: PhotoOut[];
  clusters: FaceClusterOut[];
  items: ClothingItemOut[];
}

interface HeroCard {
  id: string;
  title: string;
  image: string;
}

interface TrendingProduct {
  id: string;
  brand: string;
  name: string;
  views: number;
  image: string;
}

interface PopularSearch {
  id: string;
  label: string;
  image: string;
}

interface BrandSuggestion {
  id: string;
  label: string;
  image: string;
}

interface SearchShortcut {
  id: string;
  label: string;
  icon: keyof typeof Feather.glyphMap;
}

interface SearchBrandTile {
  id: string;
  label: string;
  image: string;
}

const MOST_RECENT_LIMIT = 10;
const POLL_MS = 1500;
const MAX_DIMENSION = 1920;
const FACE_PICKER_CIRCLE_SIZE = 72;
const CURRENT_USER_NAME = 'arjun rawal';

const HERO_CARDS: HeroCard[] = [
  {
    id: 'hero-1',
    title: "Men's Workwear Basics",
    image:
      'https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?auto=format&fit=crop&w=1200&q=80',
  },
  {
    id: 'hero-2',
    title: 'Spring Office Layers',
    image:
      'https://images.unsplash.com/photo-1487222477894-8943e31ef7b2?auto=format&fit=crop&w=1200&q=80',
  },
  {
    id: 'hero-3',
    title: 'Quiet Luxury Capsule',
    image:
      'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?auto=format&fit=crop&w=1200&q=80',
  },
];

const TRENDING_PRODUCTS: TrendingProduct[] = [
  {
    id: 'product-1',
    brand: 'EVERLANE',
    name: 'Everlane\\nBucket Bag',
    views: 221,
    image:
      'https://images.unsplash.com/photo-1584917865442-de89df76afd3?auto=format&fit=crop&w=900&q=80',
  },
  {
    id: 'product-2',
    brand: 'LINEN-BLEND DROP-WAIST',
    name: 'Linen-Blend Drop-\\nWaist Maxi Dress',
    views: 182,
    image:
      'https://images.unsplash.com/photo-1496747611176-843222e1e57c?auto=format&fit=crop&w=900&q=80',
  },
];

const TRENDING_BRANDS = [
  { id: 'brand-1', rank: '#1', label: 'REVOLVE', visits: '47.93K visits' },
  { id: 'brand-2', rank: '#2', label: 'NORDSTROM', visits: '29.98K visits' },
  { id: 'brand-3', rank: '#3', label: 'TheRealReal', visits: '27.83K visits' },
];

const POPULAR_SEARCHES: PopularSearch[] = [
  {
    id: 'search-1',
    label: 'Tb.490 Rife Shimmer Silver Sneakers',
    image:
      'https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=500&q=80',
  },
  {
    id: 'search-2',
    label: 'Ralph Lauren Cable Knit Sweater',
    image:
      'https://images.unsplash.com/photo-1611312449408-fcece27cdbb7?auto=format&fit=crop&w=500&q=80',
  },
  {
    id: 'search-3',
    label: 'Acne Studio Straight Leg-Jeans',
    image:
      'https://images.unsplash.com/photo-1542272604-787c3835535d?auto=format&fit=crop&w=500&q=80',
  },
  {
    id: 'search-4',
    label: "Arc'teryx Beta Jacket Men's Jacket",
    image:
      'https://images.unsplash.com/photo-1556906781-9a412961c28c?auto=format&fit=crop&w=500&q=80',
  },
  {
    id: 'search-5',
    label: "Levi's Trucker Jacket",
    image:
      'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?auto=format&fit=crop&w=500&q=80',
  },
];

const BRAND_SUGGESTIONS: BrandSuggestion[] = [
  {
    id: 'suggested-brand-1',
    label: 'MANGO',
    image:
      'https://images.unsplash.com/photo-1544441893-675973e31985?auto=format&fit=crop&w=700&q=80',
  },
  {
    id: 'suggested-brand-2',
    label: 'belk',
    image:
      'https://images.unsplash.com/photo-1519741497674-611481863552?auto=format&fit=crop&w=700&q=80',
  },
  {
    id: 'suggested-brand-3',
    label: 'Wrangler',
    image:
      'https://images.unsplash.com/photo-1541099649105-f69ad21f3246?auto=format&fit=crop&w=700&q=80',
  },
];

const HOME_FEED_TABS: { key: HomeFeedTab; label: string }[] = [
  { key: 'explore', label: 'Explore' },
  { key: 'forYou', label: 'For you' },
  { key: 'trending', label: 'Trending' },
];
const MAIN_TABS: MainTab[] = ['home', 'search', 'saved', 'profile'];

const SEARCH_SHORTCUTS: SearchShortcut[] = [
  { id: 'shortcut-outfits', label: 'Outfits', icon: 'shield' },
  { id: 'shortcut-upload', label: 'Upload pic', icon: 'image' },
  { id: 'shortcut-occasions', label: 'Occasions', icon: 'coffee' },
  { id: 'shortcut-sales', label: 'Sales', icon: 'tag' },
  { id: 'shortcut-favorites', label: 'Favorites', icon: 'bookmark' },
  { id: 'shortcut-luxury', label: 'Luxury', icon: 'star' },
];

const SEARCH_BRAND_TILES: SearchBrandTile[] = [
  {
    id: 'search-brand-mango',
    label: 'MANGO',
    image:
      'https://images.unsplash.com/photo-1485230895905-ec40ba36b9bc?auto=format&fit=crop&w=900&q=80',
  },
  {
    id: 'search-brand-wrangler',
    label: 'Wrangler',
    image:
      'https://images.unsplash.com/photo-1542272604-787c3835535d?auto=format&fit=crop&w=900&q=80',
  },
  {
    id: 'search-brand-belk',
    label: 'belk',
    image:
      'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?auto=format&fit=crop&w=900&q=80',
  },
  {
    id: 'search-brand-adidas',
    label: 'adidas',
    image:
      'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=900&q=80',
  },
];

const FOR_YOU_SPORT_PRODUCTS = [
  {
    id: 'sport-1',
    title: 'Nike Tech Zip Hoodie',
    price: '$79',
    image:
      'https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?auto=format&fit=crop&w=900&q=80',
  },
  {
    id: 'sport-2',
    title: 'Nike Running Shorts',
    price: '$25',
    image:
      'https://images.unsplash.com/photo-1617952236317-7f5b9db59adb?auto=format&fit=crop&w=900&q=80',
  },
  {
    id: 'sport-3',
    title: 'Performance Track Set',
    price: '$88',
    image:
      'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=900&q=80',
  },
];

function backendBaseUrl(): string {
  const configured = process.env.EXPO_PUBLIC_BACKEND_URL;
  if (configured) return configured.replace(/\/$/, '');

  if (Platform.OS === 'android') {
    return 'http://10.0.2.2:8000';
  }

  const hostUri = Constants.expoConfig?.hostUri;
  if (hostUri) {
    const host = hostUri.split(':')[0];
    return `http://${host}:8000`;
  }

  return 'http://localhost:8000';
}

const BACKEND = backendBaseUrl();

function statusLabel(status: SyncStatus): string {
  switch (status) {
    case 'requesting-permission':
      return 'Requesting camera roll permission';
    case 'reading-camera-roll':
      return 'Reading most recent photos';
    case 'uploading':
      return 'Uploading photos';
    case 'analyzing_faces':
      return 'Analyzing faces';
    case 'awaiting_face_pick':
      return 'Select your face cluster';
    case 'extracting_clothing':
      return 'Extracting clothing labels';
    case 'done':
      return 'Done';
    case 'failed':
      return 'Failed';
    default:
      return 'Idle';
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BACKEND}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status}`);
  }
  return (await res.json()) as T;
}

async function preprocessAssetUri(
  sourceUri: string,
  width: number,
  height: number,
): Promise<string> {
  const needsConversion = /\.heic$/i.test(sourceUri);
  const isLarge = width > MAX_DIMENSION || height > MAX_DIMENSION;

  if (!needsConversion && !isLarge) {
    return sourceUri;
  }

  const resizeAction = isLarge
    ? width >= height
      ? [{ resize: { width: MAX_DIMENSION } }]
      : [{ resize: { height: MAX_DIMENSION } }]
    : [];

  const processed = await manipulateAsync(sourceUri, resizeAction, {
    format: SaveFormat.JPEG,
    compress: 0.82,
  });
  return processed.uri;
}

function getFaceCropStyle(
  bb: FaceClusterOut['rep_bbox'],
  aspectRatio: number,
) {
  const safeAspectRatio = aspectRatio > 0 ? aspectRatio : 1;
  const visibleFraction = Math.max(bb.width, bb.height) * 1.8;
  const scale = FACE_PICKER_CIRCLE_SIZE / Math.max(visibleFraction, 0.0001);

  const imgW = scale * (safeAspectRatio >= 1 ? 1 : safeAspectRatio);
  const imgH = scale * (safeAspectRatio >= 1 ? 1 / safeAspectRatio : 1);

  const centerX = bb.left + bb.width / 2;
  const centerY = bb.top + bb.height / 2;

  return {
    position: 'absolute' as const,
    width: imgW,
    height: imgH,
    left: FACE_PICKER_CIRCLE_SIZE / 2 - centerX * imgW,
    top: FACE_PICKER_CIRCLE_SIZE / 2 - centerY * imgH,
  };
}

function serifStyle(isItalic = false) {
  return {
    fontFamily: Platform.select({
      ios: isItalic ? 'Times New Roman' : 'Didot',
      android: 'serif',
      default: 'serif',
    }),
    fontStyle: isItalic ? 'italic' : 'normal',
  } as const;
}

export default function Index() {
  const [status, setStatus] = useState<SyncStatus>('idle');
  const [job, setJob] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadCount, setUploadCount] = useState<{ total: number; uploaded: number }>({
    total: 0,
    uploaded: 0,
  });
  const [busySelectCluster, setBusySelectCluster] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<MainTab>('home');
  const [savedTab, setSavedTab] = useState<SavedTab>('items');
  const [searchText, setSearchText] = useState('');
  const [searchFocused, setSearchFocused] = useState(false);
  const [homeFeedTab, setHomeFeedTab] = useState<HomeFeedTab>('trending');
  const [showSyncTermsModal, setShowSyncTermsModal] = useState(false);
  const [bottomNavTrackWidth, setBottomNavTrackWidth] = useState(0);
  const bottomNavHighlightX = useRef(new Animated.Value(0)).current;
  const searchInputRef = useRef<TextInput | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bottomTabWidth = bottomNavTrackWidth > 0 ? bottomNavTrackWidth / MAIN_TABS.length : 0;

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollJob = useCallback(
    async (jobId: string) => {
      try {
        const next = await fetchJson<JobDetail>(`/api/jobs/${jobId}`);
        setJob(next);
        setStatus(next.status);
        setError(next.error ?? null);
        const pending = next.items.filter((item) => item.tier === 'pending').length;
        if (next.status === 'failed' || (next.status === 'done' && pending === 0)) {
          stopPolling();
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [stopPolling],
  );

  const beginPolling = useCallback(
    (jobId: string) => {
      stopPolling();
      void pollJob(jobId);
      pollRef.current = setInterval(() => {
        void pollJob(jobId);
      }, POLL_MS);
    },
    [pollJob, stopPolling],
  );

  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  useEffect(() => {
    if (!bottomTabWidth) return;
    const targetIndex = MAIN_TABS.indexOf(activeTab);
    Animated.spring(bottomNavHighlightX, {
      toValue: targetIndex * bottomTabWidth,
      useNativeDriver: true,
      bounciness: 0,
      speed: 16,
    }).start();
  }, [activeTab, bottomTabWidth, bottomNavHighlightX]);

  useEffect(() => {
    if (activeTab !== 'search') {
      setSearchFocused(false);
    }
  }, [activeTab]);

  useEffect(() => {
    if (!searchFocused) return;
    const timer = setTimeout(() => {
      searchInputRef.current?.focus();
    }, 80);
    return () => clearTimeout(timer);
  }, [searchFocused]);

  const syncCameraRoll = useCallback(async () => {
    try {
      setError(null);
      setJob(null);
      setStatus('requesting-permission');
      setUploadCount({ total: 0, uploaded: 0 });

      const permission = await MediaLibrary.requestPermissionsAsync();
      if (permission.status !== 'granted') {
        setStatus('failed');
        setError('Camera roll permission was denied.');
        return;
      }

      setStatus('reading-camera-roll');
      const result = await MediaLibrary.getAssetsAsync({
        first: MOST_RECENT_LIMIT,
        mediaType: [MediaLibrary.MediaType.photo],
        sortBy: [MediaLibrary.SortBy.creationTime],
      });

      if (result.assets.length === 0) {
        setStatus('failed');
        setError('No photos found in your camera roll.');
        return;
      }

      const assetsToUpload = result.assets.slice(0, MOST_RECENT_LIMIT);
      setUploadCount({ total: assetsToUpload.length, uploaded: 0 });

      setStatus('uploading');
      const form = new FormData();
      form.append('user_name', CURRENT_USER_NAME);
      let uploaded = 0;
      for (let i = 0; i < assetsToUpload.length; i += 1) {
        const asset = assetsToUpload[i];
        const info = await MediaLibrary.getAssetInfoAsync(asset);
        const local = info.localUri ?? asset.uri;
        const uri = await preprocessAssetUri(local, asset.width ?? 0, asset.height ?? 0);
        if (!uri) continue;

        form.append('photos', {
          uri,
          name: `${asset.id || `latest-photo-${i + 1}`}.jpg`,
          type: 'image/jpeg',
        } as any);
        uploaded += 1;
        setUploadCount({ total: assetsToUpload.length, uploaded });
      }

      if (uploaded === 0) {
        setStatus('failed');
        setError('Could not access local image files for upload.');
        return;
      }

      const created = await fetchJson<{ job: JobDetail }>('/api/jobs', {
        method: 'POST',
        body: form,
      });

      setJob(created.job);
      setStatus(created.job.status);
      setActiveTab('saved');
      setSavedTab('items');
      beginPolling(created.job.id);
    } catch (e) {
      setStatus('failed');
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [beginPolling]);

  const chooseCluster = useCallback(
    async (clusterId: string) => {
      if (!job) return;
      setBusySelectCluster(clusterId);
      try {
        const selected = await fetchJson<{ job: JobDetail }>(`/api/jobs/${job.id}/select-cluster`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cluster_id: clusterId }),
        });
        setJob(selected.job);
        setStatus(selected.job.status);
        beginPolling(job.id);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusySelectCluster(null);
      }
    },
    [beginPolling, job],
  );

  const groupedByPhoto = useMemo(() => {
    if (!job) return [] as { photo: PhotoOut; items: ClothingItemOut[] }[];
    const byPhoto = new Map<string, ClothingItemOut[]>();
    for (const item of job.items) {
      const arr = byPhoto.get(item.photo_id) ?? [];
      arr.push(item);
      byPhoto.set(item.photo_id, arr);
    }

    return job.photos
      .map((photo) => ({ photo, items: byPhoto.get(photo.id) ?? [] }))
      .filter((section) => section.items.length > 0);
  }, [job]);

  const counts = useMemo(() => {
    if (!job) return { exact: 0, similar: 0, pending: 0, generic: 0 };
    return {
      exact: job.items.filter((item) => item.tier === 'exact').length,
      similar: job.items.filter((item) => item.tier === 'similar').length,
      pending: job.items.filter((item) => item.tier === 'pending').length,
      generic: job.items.filter((item) => item.tier === 'generic').length,
    };
  }, [job]);

  const inFlight =
    status === 'requesting-permission' ||
    status === 'reading-camera-roll' ||
    status === 'uploading' ||
    status === 'analyzing_faces' ||
    status === 'extracting_clothing';

  const savedHeroCopy = useMemo(() => {
    if (savedTab === 'wishlists') {
      return {
        title: 'Create & share wishlists\nfor your best finds',
        action: '+ Create wishlist',
      };
    }

    if (savedTab === 'brands') {
      return {
        title: 'Get notified about deals\nfrom your favorite brands',
        action: '+ Add brands',
      };
    }

    return {
      title: 'Save your favorites,\nget notified on price drops',
      action: '+ Add items',
    };
  }, [savedTab]);

  const searchResults = useMemo(() => {
    const query = searchText.trim().toLowerCase();
    if (!query) return POPULAR_SEARCHES;
    return POPULAR_SEARCHES.filter((entry) => entry.label.toLowerCase().includes(query));
  }, [searchText]);

  const renderSyncCard = (showItems: boolean) => (
    <View style={styles.syncCard}>
      <View style={styles.syncTitleRow}>
        <Text style={styles.syncTitle}>Closet sync</Text>
        <Text style={styles.syncState}>{statusLabel(status)}</Text>
      </View>

      <Pressable
        onPress={() => setShowSyncTermsModal(true)}
        disabled={inFlight}
        style={({ pressed }) => [
          styles.syncButton,
          inFlight || pressed ? styles.syncButtonPressed : null,
        ]}
      >
        {inFlight ? <ActivityIndicator color="#fff" size="small" /> : <Feather name="refresh-cw" size={15} color="#fff" />}
        <Text style={styles.syncButtonText}>{inFlight ? 'Syncing photos...' : 'Sync camera roll'}</Text>
      </Pressable>

      {status === 'reading-camera-roll' || status === 'uploading' ? (
        <Text style={styles.syncMetaText}>
          {uploadCount.uploaded}/{uploadCount.total} prepared
        </Text>
      ) : null}

      {job ? (
        <View style={styles.syncMetaGroup}>
          <Text style={styles.syncMetaText}>
            job {job.id.slice(0, 8)} • {job.photo_count} photos • {job.items.length} items
          </Text>
          <Text style={styles.syncMetaText}>
            {counts.exact} exact • {counts.similar} similar • {counts.pending} pending • {counts.generic} no-match
          </Text>
        </View>
      ) : null}

      {error ? <Text style={styles.errorText}>{error}</Text> : null}

      {showItems && job && status === 'awaiting_face_pick' ? (
        <View style={styles.clusterPanel}>
          <Text style={styles.clusterTitle}>Pick your face</Text>
          <View style={styles.clusterGrid}>
            {job.clusters.map((cluster) => {
              const imgStyle = getFaceCropStyle(cluster.rep_bbox, cluster.rep_aspect_ratio);
              return (
                <Pressable
                  key={cluster.id}
                  onPress={() => chooseCluster(cluster.id)}
                  disabled={busySelectCluster !== null}
                  style={({ pressed }) => [
                    styles.clusterTile,
                    cluster.id === job.selected_cluster_id ? styles.clusterTileActive : null,
                    pressed ? styles.clusterTilePressed : null,
                  ]}
                >
                  <View
                    style={[
                      styles.clusterFaceCircle,
                      cluster.id === job.selected_cluster_id ? styles.clusterFaceCircleActive : null,
                    ]}
                  >
                    <Image
                      source={{ uri: cluster.source_url }}
                      style={imgStyle}
                      resizeMode="stretch"
                    />
                  </View>
                  <Text style={styles.clusterCount}>{cluster.member_count} photos</Text>
                  {busySelectCluster === cluster.id ? <ActivityIndicator size="small" color="#111" /> : null}
                </Pressable>
              );
            })}
          </View>
        </View>
      ) : null}

      {showItems && groupedByPhoto.length > 0 ? (
        <View style={styles.syncedItemsSection}>
          <Text style={styles.syncedItemsHeading}>Synced closet</Text>
          <View style={styles.syncedItemsList}>
            {groupedByPhoto.slice(0, 8).flatMap((section) =>
              section.items.slice(0, 3).map((item) => (
                <View key={item.id} style={styles.syncedItemCard}>
                  <Image
                    source={{ uri: item.crop_url ?? section.photo.url }}
                    style={styles.syncedItemImage}
                  />
                  <View style={styles.syncedItemTextWrap}>
                    <Text style={styles.syncedItemBrand} numberOfLines={1}>
                      {item.brand_visible ?? item.category.toUpperCase()}
                    </Text>
                    <Text style={styles.syncedItemTitle} numberOfLines={2}>
                      {item.description}
                    </Text>
                    {item.best_match ? (
                      <Pressable
                        onPress={() => {
                          if (item.best_match?.link) {
                            Linking.openURL(item.best_match.link).catch(() => {});
                          }
                        }}
                      >
                        <Text style={styles.bestMatchLink} numberOfLines={1}>
                          {item.best_match.title}
                        </Text>
                      </Pressable>
                    ) : (
                      <Text style={styles.bestMatchFallback}>No match yet</Text>
                    )}
                  </View>
                </View>
              )),
            )}
          </View>
        </View>
      ) : null}

      {showItems && job && status === 'done' && job.items.length === 0 ? (
        <Text style={styles.syncMetaText}>No clothing labels were found in this sync run.</Text>
      ) : null}
    </View>
  );

  const renderHomeTrendingFeed = () => (
    <View>
      <View style={styles.sectionWrap}>
        <Text style={[styles.sectionHeading, serifStyle()]}>Top trends</Text>

        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.heroScroller}>
          {HERO_CARDS.map((card) => (
            <View key={card.id} style={styles.heroCard}>
              <Image source={{ uri: card.image }} style={styles.heroImage} />
              <View style={styles.heroOverlay}>
                <Text style={[styles.heroTitle, serifStyle()]}>{card.title}</Text>
                <Pressable style={styles.heroCta}>
                  <Text style={styles.heroCtaText}>See the list</Text>
                </Pressable>
              </View>
            </View>
          ))}
        </ScrollView>

        <View style={styles.dotRow}>
          <View style={[styles.dot, styles.dotActive]} />
          <View style={styles.dot} />
          <View style={styles.dot} />
          <View style={styles.dot} />
          <View style={styles.dot} />
        </View>
      </View>

      <View style={styles.sectionWrap}>
        <View style={styles.headingRow}>
          <Text style={[styles.sectionHeading, serifStyle()]}>
            Trending on <Text style={[styles.logoInHeading, serifStyle(true)]}>phia</Text>
          </Text>
          <View style={styles.chevronCircle}>
            <Feather name="chevron-right" size={18} color="#7a7a7a" />
          </View>
        </View>
        <View style={styles.weekFilterRow}>
          <Text style={styles.weekFilterText}>This week</Text>
          <Feather name="chevron-down" size={16} color="#7f7f7f" />
        </View>
      </View>

      <View style={styles.productGrid}>
        {TRENDING_PRODUCTS.map((product) => (
          <View key={product.id} style={styles.productCell}>
            <View style={styles.productImageWrap}>
              <Image source={{ uri: product.image }} style={styles.productImage} />
              <View style={styles.viewsPill}>
                <Ionicons name="eye-outline" size={14} color="#636363" />
                <Text style={styles.viewsText}>{product.views}</Text>
              </View>
            </View>
            <Text style={styles.productBrand}>{product.brand}</Text>
            <Text style={styles.productName}>{product.name}</Text>
          </View>
        ))}
      </View>

      <View style={styles.sectionWrap}>
        <Text style={[styles.sectionHeading, serifStyle()]}>Trending brands</Text>
        <View style={styles.weekFilterRow}>
          <Text style={styles.weekFilterText}>This week</Text>
          <Feather name="chevron-down" size={16} color="#7f7f7f" />
        </View>
      </View>

      <View>
        {TRENDING_BRANDS.map((brand) => (
          <View key={brand.id} style={styles.brandRow}>
            <Text style={styles.brandRank}>{brand.rank}</Text>
            <View style={styles.brandInfo}>
              <Text style={[styles.brandName, serifStyle(brand.label.includes('Real'))]}>{brand.label}</Text>
              <Text style={styles.brandVisits}>{brand.visits}</Text>
            </View>
            <View style={styles.bookmarkCircle}>
              <Feather name="bookmark" size={18} color="#9d9d9d" />
            </View>
          </View>
        ))}
      </View>
    </View>
  );

  const renderHomeForYouFeed = () => (
    <View>
      <View style={styles.sectionWrap}>
        <Text style={[styles.sectionHeading, serifStyle()]}>Browse styles</Text>
        <View style={styles.forYouSeasonCard}>
          <Image
            source={{
              uri: 'https://images.unsplash.com/photo-1543076447-215ad9ba6923?auto=format&fit=crop&w=1000&q=80',
            }}
            style={styles.forYouSeasonImage}
          />
          <Text style={[styles.forYouSeasonLabel, serifStyle()]}>Winter</Text>
        </View>
      </View>

      <View style={styles.feedDivider} />

      <View style={styles.sectionWrap}>
        <View style={styles.headingRow}>
          <Text style={[styles.sectionHeading, serifStyle()]}>Sport Time</Text>
          <View style={styles.chevronCircle}>
            <Feather name="chevron-right" size={18} color="#7a7a7a" />
          </View>
        </View>
      </View>

      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.forYouProductsScroller}>
        {FOR_YOU_SPORT_PRODUCTS.map((item) => (
          <View key={item.id} style={styles.forYouProductCard}>
            <Image source={{ uri: item.image }} style={styles.forYouProductImage} />
            <View style={styles.forYouBookmark}>
              <Feather name="bookmark" size={18} color="#9d9d9d" />
            </View>
            <Text style={styles.forYouPrice}>{item.price}</Text>
          </View>
        ))}
      </ScrollView>
    </View>
  );

  const renderHomeExploreFeed = () => (
    <View style={styles.exploreFeedRoot}>
      <View style={styles.exploreColumns}>
        <View style={styles.exploreColumn}>
          <View style={styles.exploreBootCard}>
            <Image
              source={{
                uri: 'https://images.unsplash.com/photo-1516257984-b1b4d707412e?auto=format&fit=crop&w=900&q=80',
              }}
              style={styles.exploreBootImage}
            />
            <View style={styles.exploreBootOverlay}>
              <Text style={styles.exploreBootTitle}>Boot Cut</Text>
              <Text style={styles.exploreBootSubtitle}>20 items</Text>
            </View>
          </View>

          <View style={styles.exploreMiniRow}>
            <Image
              source={{
                uri: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=700&q=80',
              }}
              style={styles.exploreMiniWide}
            />
            <Image
              source={{
                uri: 'https://images.unsplash.com/photo-1604176424472-9d57a6f18e6c?auto=format&fit=crop&w=500&q=80',
              }}
              style={styles.exploreMiniNarrow}
            />
          </View>

          <View style={styles.exploreCreatorRow}>
            <View style={styles.exploreCreatorAvatar}>
              <Text style={[styles.exploreCreatorAvatarText, serifStyle(true)]}>phia</Text>
            </View>
            <Text style={styles.exploreCreatorLabel}>By @phiaco</Text>
            <Ionicons name="checkmark-circle" size={17} color="#5d5d62" />
          </View>

          <View style={styles.exploreSharedCard}>
            <Image
              source={{
                uri: 'https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=900&q=80',
              }}
              style={styles.exploreSharedMain}
            />
            <View style={styles.exploreSharedSideColumn}>
              <Image
                source={{
                  uri: 'https://images.unsplash.com/photo-1520975954732-35dd22cf7f8f?auto=format&fit=crop&w=400&q=80',
                }}
                style={styles.exploreSharedSideThumb}
              />
              <Image
                source={{
                  uri: 'https://images.unsplash.com/photo-1512436991641-6745cdb1723f?auto=format&fit=crop&w=400&q=80',
                }}
                style={styles.exploreSharedSideThumb}
              />
              <Image
                source={{
                  uri: 'https://images.unsplash.com/photo-1515843279827-4b82a9f5f44d?auto=format&fit=crop&w=400&q=80',
                }}
                style={styles.exploreSharedSideThumb}
              />
            </View>
          </View>
          <Text style={styles.exploreSharedLabel}>The Shared Closet</Text>
        </View>

        <View style={styles.exploreColumn}>
          <View style={styles.exploreLookCard}>
            <Image
              source={{
                uri: 'https://images.unsplash.com/photo-1594938328870-9623159c8c99?auto=format&fit=crop&w=900&q=80',
              }}
              style={styles.exploreLookImage}
            />
          </View>
          <Text style={styles.exploreLookTitle}>Refined Layering</Text>

          <View style={styles.exploreRightCard}>
            <Image
              source={{
                uri: 'https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=900&q=80',
              }}
              style={styles.exploreRightCardImage}
            />
            <View style={styles.exploreBookmarkPill}>
              <Feather name="bookmark" size={18} color="#9d9d9d" />
            </View>
          </View>
        </View>
      </View>
    </View>
  );

  const renderHome = () => (
    <View>
      <View style={styles.homeTopBar}>
        <Text style={[styles.logoText, serifStyle(true)]}>phia</Text>
        <View style={styles.homeRightControls}>
          <Pressable style={styles.winPill}>
            <Feather name="gift" size={13} color="#fff" />
            <Text style={styles.winPillText}>Win a Birkin</Text>
          </Pressable>
          <View style={styles.iconCluster}>
            <Pressable style={styles.iconButton}>
              <SimpleLineIcons name="bell" size={14} color="#121212" />
            </Pressable>
            <Pressable style={styles.iconButton} onPress={() => setActiveTab('search')}>
              <Feather name="search" size={16} color="#121212" />
            </Pressable>
          </View>
        </View>
      </View>

      <View style={styles.modeTabs}>
        {HOME_FEED_TABS.map((tab) => (
          <Pressable
            key={tab.key}
            onPress={() => setHomeFeedTab(tab.key)}
            style={styles.modeTabButton}
          >
            <Text style={homeFeedTab === tab.key ? styles.modeTabActive : styles.modeTabMuted}>
              {tab.label}
            </Text>
            <View
              style={[
                styles.modeTabUnderline,
                homeFeedTab === tab.key ? styles.modeTabUnderlineActive : null,
              ]}
            />
          </Pressable>
        ))}
      </View>

      {homeFeedTab === 'trending' ? renderHomeTrendingFeed() : null}
      {homeFeedTab === 'forYou' ? renderHomeForYouFeed() : null}
      {homeFeedTab === 'explore' ? renderHomeExploreFeed() : null}
    </View>
  );

  const renderSearch = () => {
    if (searchFocused) {
      return (
        <View style={styles.searchPage}>
          <View style={styles.searchTopRow}>
            <Pressable
              style={styles.backButton}
              onPress={() => {
                searchInputRef.current?.blur();
                setSearchFocused(false);
              }}
            >
              <Feather name="chevron-left" size={24} color="#131313" />
            </Pressable>

            <View style={styles.searchInputWrap}>
              <Feather name="search" size={17} color="#666" />
              <View style={styles.searchInputTextWrap}>
                <Text style={styles.searchHint}>Paste URL or search</Text>
                <TextInput
                  ref={searchInputRef}
                  value={searchText}
                  onChangeText={setSearchText}
                  onFocus={() => setSearchFocused(true)}
                  placeholder="Search for an item"
                  placeholderTextColor="#666"
                  style={styles.searchInput}
                />
              </View>
            </View>
          </View>

          <Text style={[styles.searchTitle, serifStyle()]}>Popular searches</Text>

          <View style={styles.searchResults}>
            {searchResults.map((entry) => (
              <Pressable key={entry.id} style={styles.searchRow}>
                <Image source={{ uri: entry.image }} style={styles.searchThumb} />
                <Text style={styles.searchRowLabel}>{entry.label}</Text>
                <Feather name="chevron-right" size={22} color="#606060" />
              </Pressable>
            ))}
          </View>
        </View>
      );
    }

    return (
      <View style={styles.searchPage}>
        <View style={styles.searchDiscoveryTopRow}>
          <Pressable
            style={styles.searchDiscoveryBar}
            onPress={() => {
              setSearchFocused(true);
            }}
          >
            <Feather name="search" size={24} color="#66666c" />
            <Text style={styles.searchDiscoveryText}>Paste URL or search</Text>
          </Pressable>
          <Pressable style={styles.searchDiscoveryCamera}>
            <Feather name="camera" size={24} color="#69696e" />
          </Pressable>
        </View>

        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.searchShortcutRow}
        >
          {SEARCH_SHORTCUTS.map((shortcut) => (
            <Pressable key={shortcut.id} style={styles.searchShortcutItem}>
              <Feather name={shortcut.icon} size={26} color="#66666c" />
              <Text style={styles.searchShortcutLabel}>{shortcut.label}</Text>
            </Pressable>
          ))}
        </ScrollView>

        <View style={styles.searchSectionCard}>
          <Text style={[styles.searchSectionTitle, serifStyle()]}>Search by look</Text>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.searchLookScroller}
          >
            {HERO_CARDS.map((card) => (
              <View key={card.id} style={styles.searchLookCard}>
                <Image source={{ uri: card.image }} style={styles.searchLookImage} />
                <View style={styles.heroOverlay}>
                  <Text style={[styles.heroTitle, serifStyle()]}>{card.title}</Text>
                  <Pressable style={styles.heroCta}>
                    <Text style={styles.heroCtaText}>See the list</Text>
                  </Pressable>
                </View>
              </View>
            ))}
          </ScrollView>
        </View>

        <View style={styles.searchSectionCard}>
          <View style={styles.searchSectionHeaderRow}>
            <Text style={[styles.searchSectionTitle, serifStyle()]}>Search by brand</Text>
            <View style={styles.searchBrandChevron}>
              <Feather name="chevron-right" size={24} color="#76767b" />
            </View>
          </View>
          <View style={styles.searchBrandGrid}>
            {SEARCH_BRAND_TILES.map((brand) => (
              <View key={brand.id} style={styles.searchBrandTile}>
                <Image source={{ uri: brand.image }} style={styles.searchBrandImage} />
                <View style={styles.searchBrandOverlay} />
                <Text style={styles.searchBrandText}>{brand.label}</Text>
              </View>
            ))}
          </View>
        </View>
      </View>
    );
  };

  const renderSaved = () => (
    <View style={styles.savedPage}>
      <View style={styles.savedHeaderRow}>
        <Text style={[styles.savedTitle, serifStyle()]}>Your saved</Text>
        <View style={styles.addCircle}>
          <Feather name="plus" size={28} color="#97979d" />
        </View>
      </View>

      <View style={styles.segmentedWrap}>
        {(['wishlists', 'items', 'brands'] as SavedTab[]).map((tab) => (
          <Pressable
            key={tab}
            onPress={() => setSavedTab(tab)}
            style={[styles.segmentedItem, savedTab === tab ? styles.segmentedItemActive : null]}
          >
            <Text style={savedTab === tab ? styles.segmentedTextActive : styles.segmentedText}>
              {tab === 'wishlists' ? 'Wishlists' : tab === 'items' ? 'Items' : 'Brands'}
            </Text>
          </Pressable>
        ))}
      </View>

      <View style={styles.savedHeroCard}>
        <Text style={[styles.savedHeroTitle, serifStyle()]}>{savedHeroCopy.title}</Text>
        <Pressable style={styles.savedHeroButton}>
          <Text style={styles.savedHeroButtonText}>{savedHeroCopy.action}</Text>
        </Pressable>
      </View>

      {savedTab === 'wishlists' ? (
        <View style={styles.savedSection}>
          <Text style={[styles.savedSectionTitle, serifStyle()]}>Editor&apos;s picks</Text>
          <View style={styles.editorCard}>
            <Image
              source={{
                uri: 'https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?auto=format&fit=crop&w=1200&q=80',
              }}
              style={styles.editorImage}
            />
            <View style={styles.editorOverlay}>
              <Text style={[styles.editorTitle, serifStyle()]}>Men&apos;s Workwear Basics</Text>
              <Pressable style={styles.heroCta}>
                <Text style={styles.heroCtaText}>Shop the list</Text>
              </Pressable>
            </View>
          </View>
        </View>
      ) : null}

      {savedTab === 'items' ? (
        <View style={styles.savedSection}>
          <View style={styles.savedSectionRow}>
            <Text style={[styles.savedSectionTitle, serifStyle()]}>Previously viewed</Text>
            <View style={styles.chevronCircle}>
              <Feather name="chevron-right" size={18} color="#7a7a7a" />
            </View>
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.placeholderCardRow}>
            {[0, 1, 2].map((index) => (
              <View key={index} style={styles.placeholderProduct}>
                <View style={styles.placeholderBookmark}>
                  <Feather name="bookmark" size={20} color="#95959a" />
                </View>
              </View>
            ))}
          </ScrollView>

          {renderSyncCard(true)}
        </View>
      ) : null}

      {savedTab === 'brands' ? (
        <View style={styles.savedSection}>
          <View style={styles.savedSectionRow}>
            <Text style={[styles.savedSectionTitle, serifStyle()]}>Brands you may like</Text>
            <View style={styles.chevronCircle}>
              <Feather name="chevron-right" size={18} color="#7a7a7a" />
            </View>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.brandSuggestionRow}>
            {BRAND_SUGGESTIONS.map((entry) => (
              <View key={entry.id} style={styles.brandSuggestionCard}>
                <Image source={{ uri: entry.image }} style={styles.brandSuggestionImage} />
                <View style={styles.brandSuggestionOverlay} />
                <View style={styles.placeholderBookmark}>
                  <Feather name="bookmark" size={20} color="#95959a" />
                </View>
                <Text style={styles.brandSuggestionLabel}>{entry.label}</Text>
              </View>
            ))}
          </ScrollView>
        </View>
      ) : null}
    </View>
  );

  const renderProfileCard = (
    title: string,
    subtitle: string,
    icon: keyof typeof Ionicons.glyphMap,
    smallPill?: string,
  ) => (
    <View style={styles.profileUtilityCard}>
      <Text style={styles.profileUtilityTitle}>{title}</Text>
      <Text style={styles.profileUtilitySubtitle}>{subtitle}</Text>
      {smallPill ? (
        <View style={styles.utilityPill}>
          <Text style={styles.utilityPillText}>{smallPill}</Text>
        </View>
      ) : null}
      <View style={styles.utilityIconWrap}>
        <Ionicons name={icon} size={38} color="#cacacf" />
      </View>
    </View>
  );

  const renderProfile = () => (
    <View style={styles.profilePage}>
      <View style={styles.profileTopRow}>
        <View style={styles.helpCircle}>
          <Feather name="help-circle" size={24} color="#111" />
        </View>
        <View style={styles.helpCircle}>
          <Feather name="settings" size={22} color="#111" />
        </View>
      </View>

      <View style={styles.profileAvatarWrap}>
        <View style={styles.avatarPill}>
          <Text style={[styles.avatarText, serifStyle()]}>AR</Text>
        </View>
        <Text style={[styles.profileName, serifStyle()]}>{CURRENT_USER_NAME}</Text>
        <Pressable style={styles.editProfileButton}>
          <Text style={styles.editProfileText}>Edit profile</Text>
        </Pressable>
      </View>

      <View style={styles.utilityGrid}>
        {renderProfileCard('Price drop', 'alerts', 'notifications-outline', 'No updates')}
        {renderProfileCard('Your link', 'history', 'attach-outline', '• 7 new')}
      </View>
      <View style={styles.utilityGrid}>
        {renderProfileCard('Edit your', 'brands', 'albums-outline')}
        {renderProfileCard('Gender', 'preferences', 'person-outline')}
      </View>

      <View style={styles.feedbackCard}>
        <Text style={styles.feedbackText}>Give feedback</Text>
        <Pressable style={styles.feedbackButton}>
          <Text style={styles.feedbackButtonText}>Text us</Text>
        </Pressable>
      </View>

      {renderSyncCard(false)}
    </View>
  );

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.appFrame}>
        <ScrollView
          style={styles.mainScroll}
          contentContainerStyle={styles.mainScrollContent}
          showsVerticalScrollIndicator={false}
        >
          {activeTab === 'home' ? renderHome() : null}
          {activeTab === 'search' ? renderSearch() : null}
          {activeTab === 'saved' ? renderSaved() : null}
          {activeTab === 'profile' ? renderProfile() : null}
        </ScrollView>

        <View style={styles.bottomNavShell}>
          <View
            style={styles.bottomNavTrack}
            onLayout={(event) => {
              setBottomNavTrackWidth(event.nativeEvent.layout.width);
            }}
          >
            {bottomTabWidth > 0 ? (
              <Animated.View
                pointerEvents="none"
                style={[
                  styles.bottomNavHighlight,
                  {
                    width: bottomTabWidth,
                    transform: [{ translateX: bottomNavHighlightX }],
                  },
                ]}
              />
            ) : null}

            {MAIN_TABS.map((tab) => (
              <Pressable
                key={tab}
                onPress={() => setActiveTab(tab)}
                style={styles.bottomNavItem}
              >
                {tab === 'home' ? (
                  <Feather name="home" size={29} color={tab === activeTab ? '#111' : '#8f8f94'} />
                ) : null}
                {tab === 'search' ? (
                  <Feather name="search" size={29} color={tab === activeTab ? '#111' : '#8f8f94'} />
                ) : null}
                {tab === 'saved' ? (
                  <Feather name="bookmark" size={25} color={tab === activeTab ? '#111' : '#8f8f94'} />
                ) : null}
                {tab === 'profile' ? (
                  <MaterialCommunityIcons
                    name="account-circle-outline"
                    size={31}
                    color={tab === activeTab ? '#111' : '#8f8f94'}
                  />
                ) : null}
              </Pressable>
            ))}
          </View>
        </View>
      </View>

      <Modal
        visible={showSyncTermsModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowSyncTermsModal(false)}
      >
        <View style={styles.termsBackdrop}>
          <View style={styles.termsCard}>
            <Text style={styles.termsTitle}>Sync camera roll</Text>
            <Text style={styles.termsBody}>
              By continuing, phia will request iOS photo permission and upload your {MOST_RECENT_LIMIT} most
              recent photos to extract clothing insights and personalize your shopping experience.
            </Text>
            <Text style={styles.termsFootnote}>You can revoke photo access any time in iOS Settings.</Text>

            <View style={styles.termsButtonRow}>
              <Pressable style={styles.termsSecondaryButton} onPress={() => setShowSyncTermsModal(false)}>
                <Text style={styles.termsSecondaryButtonText}>Not now</Text>
              </Pressable>
              <Pressable
                style={styles.termsPrimaryButton}
                onPress={() => {
                  setShowSyncTermsModal(false);
                  void syncCameraRoll();
                }}
              >
                <Text style={styles.termsPrimaryButtonText}>Accept</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#efeff2',
  },
  appFrame: {
    flex: 1,
    backgroundColor: '#efeff2',
  },
  mainScroll: {
    flex: 1,
  },
  mainScrollContent: {
    paddingHorizontal: 14,
    paddingBottom: 120,
  },
  homeTopBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 6,
    justifyContent: 'space-between',
  },
  homeRightControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 8,
  },
  logoText: {
    fontSize: 24,
    color: '#111',
    marginTop: 1,
  },
  winPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#11141a',
    borderRadius: 22,
    borderWidth: 1,
    borderColor: '#3e4148',
    paddingHorizontal: 9,
    height: 40,
    gap: 6,
    justifyContent: 'center',
  },
  winPillText: {
    color: '#fff',
    fontSize: 11.5,
    fontWeight: '600',
  },
  iconCluster: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f5f6',
    borderRadius: 22,
    paddingHorizontal: 6,
    height: 42,
    gap: 0,
  },
  iconButton: {
    width: 30,
    height: 30,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modeTabs: {
    marginTop: 12,
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#d4d4d8',
    marginHorizontal: -14,
    paddingHorizontal: 14,
  },
  modeTabButton: {
    paddingBottom: 7,
    minWidth: 82,
    alignItems: 'center',
  },
  modeTabMuted: {
    fontSize: 12,
    color: '#7f7f84',
    fontWeight: '400',
  },
  modeTabActive: {
    fontSize: 12,
    color: '#111',
    fontWeight: '600',
  },
  modeTabUnderline: {
    marginTop: 7,
    width: 72,
    height: 4,
    borderRadius: 999,
    backgroundColor: 'transparent',
  },
  modeTabUnderlineActive: {
    backgroundColor: '#111',
  },
  sectionWrap: {
    marginTop: 16,
    marginBottom: 4,
  },
  sectionHeading: {
    fontSize: 19,
    color: '#161616',
    letterSpacing: -0.3,
  },
  heroScroller: {
    gap: 12,
    paddingTop: 10,
    paddingRight: 10,
  },
  heroCard: {
    width: 306,
    height: 244,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#ddd',
  },
  heroImage: {
    width: '100%',
    height: '100%',
  },
  heroOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'flex-end',
    padding: 12,
    backgroundColor: 'rgba(0,0,0,0.2)',
    gap: 8,
  },
  heroTitle: {
    fontSize: 17,
    color: '#f8f8f8',
    lineHeight: 26,
  },
  heroCta: {
    alignSelf: 'flex-start',
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.8)',
    paddingHorizontal: 14,
    paddingVertical: 6,
    backgroundColor: 'rgba(0,0,0,0.14)',
  },
  heroCtaText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  dotRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    marginTop: 10,
  },
  dot: {
    width: 9,
    height: 9,
    borderRadius: 4.5,
    backgroundColor: '#b6b6b8',
  },
  dotActive: {
    backgroundColor: '#111',
  },
  headingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },
  logoInHeading: {
    fontStyle: 'italic',
  },
  chevronCircle: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: '#e8e8eb',
    alignItems: 'center',
    justifyContent: 'center',
  },
  weekFilterRow: {
    marginTop: 2,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  weekFilterText: {
    fontSize: 13,
    color: '#7f7f84',
    fontWeight: '500',
  },
  productGrid: {
    flexDirection: 'row',
    marginHorizontal: -14,
    borderTopWidth: 1,
    borderTopColor: '#d2d2d5',
    borderBottomWidth: 1,
    borderBottomColor: '#d2d2d5',
  },
  productCell: {
    width: '50%',
    borderRightWidth: 1,
    borderRightColor: '#d2d2d5',
    paddingHorizontal: 10,
    paddingTop: 10,
    paddingBottom: 14,
  },
  productImageWrap: {
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#ddd',
    aspectRatio: 0.84,
    marginBottom: 8,
  },
  productImage: {
    width: '100%',
    height: '100%',
  },
  viewsPill: {
    position: 'absolute',
    left: 8,
    bottom: 8,
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderRadius: 999,
    paddingHorizontal: 9,
    paddingVertical: 4,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  viewsText: {
    fontSize: 13,
    color: '#5b5b5f',
    fontWeight: '500',
  },
  productBrand: {
    color: '#adadb2',
    fontSize: 11,
    letterSpacing: 1.3,
    textAlign: 'center',
    marginBottom: 6,
  },
  productName: {
    color: '#0f0f10',
    fontSize: 13,
    lineHeight: 20,
    textAlign: 'center',
    fontWeight: '500',
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: -14,
    paddingHorizontal: 16,
    borderTopWidth: 1,
    borderTopColor: '#d2d2d5',
    height: 94,
  },
  brandRank: {
    fontSize: 19,
    color: '#5c5c60',
    width: 36,
  },
  brandInfo: {
    flex: 1,
    marginLeft: 8,
  },
  brandName: {
    fontSize: 20,
    color: '#101011',
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  brandVisits: {
    fontSize: 12,
    color: '#5d5d61',
    marginTop: 2,
    fontWeight: '500',
  },
  bookmarkCircle: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: '#ececee',
    alignItems: 'center',
    justifyContent: 'center',
  },
  feedDivider: {
    marginHorizontal: -14,
    borderTopWidth: 1,
    borderTopColor: '#d8d8dd',
    marginTop: 8,
  },
  forYouSeasonCard: {
    marginTop: 12,
    width: 158,
    height: 286,
    borderTopLeftRadius: 80,
    borderTopRightRadius: 80,
    borderBottomLeftRadius: 14,
    borderBottomRightRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#d8d8dd',
    justifyContent: 'center',
  },
  forYouSeasonImage: {
    width: '100%',
    height: '100%',
  },
  forYouSeasonLabel: {
    position: 'absolute',
    left: 18,
    bottom: 118,
    color: '#f3f4f5',
    fontSize: 28 / 2,
  },
  forYouProductsScroller: {
    gap: 10,
    paddingRight: 10,
    marginTop: 2,
  },
  forYouProductCard: {
    width: 158,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#d7d8dc',
  },
  forYouProductImage: {
    width: '100%',
    height: 252,
  },
  forYouBookmark: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#f0f0f3',
    alignItems: 'center',
    justifyContent: 'center',
  },
  forYouPrice: {
    fontSize: 17,
    color: '#111',
    paddingHorizontal: 10,
    paddingTop: 8,
    paddingBottom: 12,
  },
  exploreFeedRoot: {
    marginTop: 10,
  },
  exploreColumns: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-start',
  },
  exploreColumn: {
    flex: 1,
    gap: 8,
  },
  exploreBootCard: {
    height: 192,
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: '#d5d5d9',
  },
  exploreBootImage: {
    width: '100%',
    height: '100%',
  },
  exploreBootOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'flex-end',
    padding: 10,
    backgroundColor: 'rgba(0,0,0,0.16)',
  },
  exploreBootTitle: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '500',
  },
  exploreBootSubtitle: {
    color: '#e8e8ea',
    fontSize: 13,
    marginTop: 2,
  },
  exploreMiniRow: {
    flexDirection: 'row',
    gap: 6,
    height: 122,
  },
  exploreMiniWide: {
    flex: 0.68,
    borderRadius: 10,
    backgroundColor: '#ddd',
  },
  exploreMiniNarrow: {
    flex: 0.32,
    borderRadius: 10,
    backgroundColor: '#ddd',
  },
  exploreCreatorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 2,
  },
  exploreCreatorAvatar: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#2e5db6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  exploreCreatorAvatarText: {
    color: '#fff',
    fontSize: 9,
  },
  exploreCreatorLabel: {
    color: '#66666b',
    fontSize: 12,
    fontWeight: '500',
  },
  exploreSharedCard: {
    height: 242,
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: '#d8d8dd',
    flexDirection: 'row',
    gap: 6,
    padding: 6,
  },
  exploreSharedMain: {
    flex: 0.77,
    borderRadius: 8,
    backgroundColor: '#ddd',
  },
  exploreSharedSideColumn: {
    flex: 0.23,
    gap: 6,
  },
  exploreSharedSideThumb: {
    flex: 1,
    borderRadius: 8,
    backgroundColor: '#ddd',
  },
  exploreSharedLabel: {
    color: '#111',
    fontSize: 18,
    marginTop: 2,
  },
  exploreLookCard: {
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: '#d8d8dd',
    height: 336,
  },
  exploreLookImage: {
    width: '100%',
    height: '100%',
  },
  exploreLookTitle: {
    color: '#111',
    fontSize: 19,
    marginTop: 2,
  },
  exploreRightCard: {
    marginTop: 8,
    height: 336,
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: '#d8d8dd',
  },
  exploreRightCardImage: {
    width: '100%',
    height: '100%',
  },
  exploreBookmarkPill: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#f0f0f3',
    alignItems: 'center',
    justifyContent: 'center',
  },
  searchPage: {
    marginTop: 2,
  },
  searchDiscoveryTopRow: {
    marginTop: 4,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  searchDiscoveryBar: {
    flex: 1,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#f3f3f5',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 16,
  },
  searchDiscoveryText: {
    color: '#606066',
    fontSize: 18,
    fontWeight: '500',
  },
  searchDiscoveryCamera: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#f3f3f5',
    alignItems: 'center',
    justifyContent: 'center',
  },
  searchShortcutRow: {
    marginTop: 14,
    paddingHorizontal: 4,
    gap: 16,
    paddingRight: 18,
  },
  searchShortcutItem: {
    width: 74,
    alignItems: 'center',
    gap: 8,
  },
  searchShortcutLabel: {
    color: '#59595f',
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
  },
  searchSectionCard: {
    marginHorizontal: -14,
    marginTop: 14,
    backgroundColor: '#f4f4f6',
    borderTopWidth: 1,
    borderTopColor: '#e2e2e6',
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  searchSectionTitle: {
    fontSize: 24 / 1.2,
    color: '#242427',
  },
  searchLookScroller: {
    gap: 10,
    paddingTop: 10,
    paddingRight: 8,
  },
  searchLookCard: {
    width: 290,
    height: 250,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#d9d9dc',
  },
  searchLookImage: {
    width: '100%',
    height: '100%',
  },
  searchSectionHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  searchBrandChevron: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: '#ececee',
    alignItems: 'center',
    justifyContent: 'center',
  },
  searchBrandGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  searchBrandTile: {
    width: '48.8%',
    height: 118,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#d8d8dc',
    justifyContent: 'center',
    alignItems: 'center',
  },
  searchBrandImage: {
    ...StyleSheet.absoluteFillObject,
    width: '100%',
    height: '100%',
  },
  searchBrandOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(20,20,24,0.20)',
  },
  searchBrandText: {
    color: '#fff',
    fontSize: 28 / 1.3,
    fontWeight: '700',
    letterSpacing: 0.4,
    textTransform: 'none',
  },
  searchTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  backButton: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f5f5f6',
  },
  searchInputWrap: {
    flex: 1,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#f5f5f6',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    gap: 8,
  },
  searchInputTextWrap: {
    flex: 1,
  },
  searchHint: {
    color: '#c9c9ce',
    fontSize: 16,
    lineHeight: 17,
    marginBottom: -2,
  },
  searchInput: {
    color: '#666',
    fontSize: 17,
    fontWeight: '500',
    paddingVertical: 0,
    paddingHorizontal: 0,
  },
  searchTitle: {
    marginTop: 18,
    fontSize: 22,
    color: '#141415',
  },
  searchResults: {
    marginTop: 8,
    gap: 6,
  },
  searchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    minHeight: 64,
  },
  searchThumb: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#ddd',
  },
  searchRowLabel: {
    flex: 1,
    fontSize: 14,
    color: '#111',
    fontWeight: '500',
  },
  savedPage: {
    marginTop: 2,
  },
  savedHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  savedTitle: {
    fontSize: 24,
    color: '#151517',
    marginLeft: 2,
  },
  addCircle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#f0f0f2',
    alignItems: 'center',
    justifyContent: 'center',
  },
  segmentedWrap: {
    marginTop: 12,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#f5f5f6',
    flexDirection: 'row',
    padding: 2,
  },
  segmentedItem: {
    flex: 1,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  segmentedItemActive: {
    backgroundColor: '#d7d7db',
  },
  segmentedText: {
    fontSize: 13,
    color: '#9a9aa0',
    fontWeight: '500',
  },
  segmentedTextActive: {
    fontSize: 13,
    color: '#111',
    fontWeight: '600',
  },
  savedHeroCard: {
    marginTop: 12,
    backgroundColor: '#f5f5f6',
    borderRadius: 22,
    paddingHorizontal: 16,
    paddingVertical: 18,
    alignItems: 'center',
    gap: 14,
  },
  savedHeroTitle: {
    fontSize: 20,
    color: '#535357',
    textAlign: 'center',
    lineHeight: 31,
  },
  savedHeroButton: {
    borderRadius: 14,
    backgroundColor: '#000',
    height: 44,
    minWidth: 170,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
  },
  savedHeroButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  savedSection: {
    marginTop: 12,
    gap: 10,
  },
  savedSectionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  savedSectionTitle: {
    fontSize: 21 / 1.2,
    color: '#141415',
  },
  placeholderCardRow: {
    gap: 10,
    paddingRight: 10,
  },
  placeholderProduct: {
    width: 130,
    height: 196,
    borderRadius: 12,
    backgroundColor: '#d8d8dc',
    position: 'relative',
  },
  placeholderBookmark: {
    position: 'absolute',
    right: 8,
    top: 8,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(245,245,247,0.96)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  editorCard: {
    width: '100%',
    borderRadius: 14,
    overflow: 'hidden',
    height: 286,
    backgroundColor: '#d9d9dc',
  },
  editorImage: {
    width: '100%',
    height: '100%',
  },
  editorOverlay: {
    ...StyleSheet.absoluteFillObject,
    padding: 12,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(0,0,0,0.18)',
    gap: 8,
  },
  editorTitle: {
    color: '#f7f7f8',
    fontSize: 22,
    lineHeight: 30,
  },
  syncCard: {
    borderRadius: 16,
    backgroundColor: '#f5f5f6',
    padding: 12,
    gap: 9,
    marginTop: 6,
  },
  syncTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  syncTitle: {
    fontSize: 12,
    color: '#111',
    fontWeight: '700',
    textTransform: 'lowercase',
  },
  syncState: {
    fontSize: 11,
    color: '#68686c',
    fontWeight: '600',
  },
  syncButton: {
    height: 42,
    borderRadius: 12,
    backgroundColor: '#111',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 10,
  },
  syncButtonPressed: {
    opacity: 0.78,
  },
  syncButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  termsBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.48)',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  termsCard: {
    width: '100%',
    maxWidth: 420,
    borderRadius: 16,
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 14,
    gap: 10,
  },
  termsTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#121212',
  },
  termsBody: {
    fontSize: 14,
    lineHeight: 21,
    color: '#2f2f34',
  },
  termsFootnote: {
    fontSize: 12,
    color: '#6d6d74',
  },
  termsButtonRow: {
    marginTop: 4,
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 10,
  },
  termsSecondaryButton: {
    height: 38,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#d7d7db',
    paddingHorizontal: 14,
    justifyContent: 'center',
  },
  termsSecondaryButtonText: {
    color: '#36363a',
    fontSize: 13,
    fontWeight: '600',
  },
  termsPrimaryButton: {
    height: 38,
    borderRadius: 10,
    backgroundColor: '#111',
    paddingHorizontal: 16,
    justifyContent: 'center',
  },
  termsPrimaryButtonText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
  },
  syncMetaText: {
    color: '#67676c',
    fontSize: 12,
    fontWeight: '500',
  },
  syncMetaGroup: {
    gap: 3,
  },
  errorText: {
    color: '#ad1818',
    fontSize: 12,
    fontWeight: '500',
  },
  clusterPanel: {
    gap: 8,
    marginTop: 6,
  },
  clusterTitle: {
    fontSize: 14,
    color: '#131313',
    fontWeight: '600',
  },
  clusterGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    paddingTop: 2,
  },
  clusterTile: {
    width: '22.5%',
    borderRadius: 10,
    backgroundColor: '#ececef',
    padding: 6,
    gap: 6,
    alignItems: 'center',
  },
  clusterTileActive: {
    backgroundColor: '#e5e5e8',
  },
  clusterTilePressed: {
    opacity: 0.8,
  },
  clusterFaceCircle: {
    width: FACE_PICKER_CIRCLE_SIZE,
    height: FACE_PICKER_CIRCLE_SIZE,
    borderRadius: FACE_PICKER_CIRCLE_SIZE / 2,
    borderWidth: 2,
    borderColor: '#8d8d95',
    overflow: 'hidden',
    backgroundColor: '#d6d6dc',
  },
  clusterFaceCircleActive: {
    borderColor: '#111',
  },
  clusterCount: {
    fontSize: 11,
    color: '#2a2a2d',
    fontWeight: '600',
    textAlign: 'center',
  },
  syncedItemsSection: {
    marginTop: 2,
    gap: 8,
  },
  syncedItemsHeading: {
    fontSize: 14,
    color: '#151515',
    fontWeight: '700',
  },
  syncedItemsList: {
    gap: 8,
  },
  syncedItemCard: {
    backgroundColor: '#ececef',
    borderRadius: 12,
    overflow: 'hidden',
    flexDirection: 'row',
    minHeight: 78,
  },
  syncedItemImage: {
    width: 78,
    height: '100%',
    backgroundColor: '#dbdbe0',
  },
  syncedItemTextWrap: {
    flex: 1,
    paddingHorizontal: 8,
    paddingVertical: 7,
    gap: 2,
  },
  syncedItemBrand: {
    color: '#7f7f84',
    fontSize: 10,
    letterSpacing: 1,
  },
  syncedItemTitle: {
    color: '#161617',
    fontSize: 12,
    lineHeight: 15,
    fontWeight: '600',
  },
  bestMatchLink: {
    color: '#203f85',
    fontSize: 11,
    fontWeight: '600',
  },
  bestMatchFallback: {
    color: '#7f7f83',
    fontSize: 11,
  },
  brandSuggestionRow: {
    gap: 10,
    paddingRight: 8,
  },
  brandSuggestionCard: {
    width: 132,
    height: 220,
    borderRadius: 12,
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: '#d6d6da',
    justifyContent: 'flex-end',
    padding: 10,
  },
  brandSuggestionImage: {
    ...StyleSheet.absoluteFillObject,
    width: '100%',
    height: '100%',
  },
  brandSuggestionOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(130,130,136,0.46)',
  },
  brandSuggestionLabel: {
    color: '#fff',
    fontSize: 22,
    fontWeight: '700',
    letterSpacing: 0.5,
    textTransform: 'none',
  },
  profilePage: {
    marginTop: 2,
    gap: 10,
  },
  profileTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 2,
  },
  helpCircle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f4f4f6',
  },
  profileAvatarWrap: {
    alignItems: 'center',
    gap: 8,
    marginBottom: 2,
  },
  avatarPill: {
    width: 98,
    height: 114,
    borderTopLeftRadius: 50,
    borderTopRightRadius: 50,
    borderBottomLeftRadius: 18,
    borderBottomRightRadius: 18,
    backgroundColor: '#304e9d',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: '#fff',
    fontSize: 44,
    marginTop: -6,
  },
  profileName: {
    fontSize: 22,
    color: '#111',
    marginTop: 2,
  },
  editProfileButton: {
    marginTop: 0,
    borderRadius: 14,
    backgroundColor: '#e3e3e5',
    paddingHorizontal: 22,
    height: 38,
    justifyContent: 'center',
    alignItems: 'center',
  },
  editProfileText: {
    color: '#6a6a70',
    fontSize: 14,
    fontWeight: '500',
  },
  utilityGrid: {
    flexDirection: 'row',
    gap: 8,
  },
  profileUtilityCard: {
    flex: 1,
    borderRadius: 12,
    backgroundColor: '#f5f5f6',
    paddingHorizontal: 10,
    paddingTop: 10,
    paddingBottom: 8,
    minHeight: 94,
    position: 'relative',
    overflow: 'hidden',
  },
  profileUtilityTitle: {
    fontSize: 12,
    color: '#46464b',
    fontWeight: '500',
  },
  profileUtilitySubtitle: {
    fontSize: 12,
    color: '#46464b',
    fontWeight: '500',
  },
  utilityPill: {
    marginTop: 8,
    backgroundColor: '#e9e9eb',
    borderRadius: 8,
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  utilityPillText: {
    color: '#b1b1b6',
    fontSize: 10,
    fontWeight: '600',
  },
  utilityIconWrap: {
    position: 'absolute',
    right: 8,
    bottom: 6,
  },
  feedbackCard: {
    marginTop: 2,
    borderRadius: 14,
    backgroundColor: '#293f84',
    minHeight: 64,
    paddingHorizontal: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  feedbackText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  feedbackButton: {
    height: 44,
    borderRadius: 22,
    minWidth: 96,
    paddingHorizontal: 14,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.75)',
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  feedbackButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
  bottomNavShell: {
    position: 'absolute',
    left: 14,
    right: 14,
    bottom: 10,
    height: 68,
    borderRadius: 34,
    backgroundColor: 'rgba(243,243,245,0.95)',
    borderWidth: 1,
    borderColor: '#e2e2e6',
    padding: 2,
  },
  bottomNavTrack: {
    flex: 1,
    flexDirection: 'row',
    borderRadius: 32,
    overflow: 'hidden',
    position: 'relative',
  },
  bottomNavHighlight: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    borderRadius: 32,
    backgroundColor: '#d8d8dc',
  },
  bottomNavItem: {
    flex: 1,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
