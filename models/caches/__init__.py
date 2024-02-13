from . import cache, kv_cache, kv_cache_train

CACHES = {
    "none": cache.LMCache,
    "kv": kv_cache.KVLMCache,
    "kv_train": kv_cache_train.KVLMCache
}


def get_cache(cache_name):
    return CACHES[cache_name]


def registered_caches():
    return CACHES.keys()
