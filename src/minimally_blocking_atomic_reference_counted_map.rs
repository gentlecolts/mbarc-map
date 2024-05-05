use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::{Arc, atomic, Mutex, MutexGuard};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::fixed_address_continuous_allocation::{FaVec, FaVecIndex};

type HashType<T, U> = HashMap<T, U>;
type SharedDataContainerType<T, const BLOCK_SIZE: usize> = Arc<Mutex<FaVec<DataHolder<T, BLOCK_SIZE>, BLOCK_SIZE>>>;

struct DataHolder<T, const BLOCK_SIZE: usize> {
	ref_count: AtomicUsize,
	pending_removal: AtomicBool,

	//TODO: having these here rather than in DataReference means less duplication, but also makes for fatter entries in the FaVec array
	//the deduplication is probably worth it, but will need to bench for it being worth the potentially less good caching behavior
	owner: SharedDataContainerType<T, BLOCK_SIZE>,
	owning_key: FaVecIndex,

	pub data: Mutex<T>,
}

impl<T, const BLOCK_SIZE: usize> DataHolder<T, BLOCK_SIZE> {
	//TODO: evaluate safety of this ordering
	fn deleted(&self) -> bool {
		self.pending_removal.load(Ordering::Acquire)
	}

	fn ref_count(&self) -> usize {
		//TODO: evaluate safety of this ordering
		self.ref_count.load(Ordering::Acquire)
	}

	fn set_deleted(&self, state: bool) {
		//TODO: evaluate safety of this ordering
		self.pending_removal.store(state, Ordering::Release);
	}
}

/// Atomic, reference counted pointer to data stored within an [MbarcMap].
///
/// A valid copy of DataReference should imply valid data being pointed to, regardless of the status of the owning [MbarcMap].
///
/// # Examples
/// ```
/// //In this snippet, Arc isn't really necessary, however it's generally recommended to wrap MbarcMap in an Arc as threading is its main use case
/// //Very important to note that you do not need to wrap MbarcMap in a Mutex, and doing so is redundant
/// use std::sync::{Arc, MutexGuard};
/// use mbarc_map::{DataReference, MbarcMap};
///
/// let concurrent_map = Arc::new(MbarcMap::<i64, &str>::new());
///
/// let key: i64 = 2;
/// let value: &str = "Hi";
/// concurrent_map.insert(key, value);
///
/// //Retrieve the item we just put into the map, then drop the map itself
/// let first_value = concurrent_map.get(&key);
/// drop(concurrent_map);
///
/// //Actual data reference
/// let first_value = first_value.unwrap();
///
/// //Since the map no longer exists, the only reference left is the one we're still holding
/// assert_eq!(first_value.ref_count(), 1);
/// //When the map is dropped all values within are marked as deleted (also the case if the individual item had been removed instead)
/// assert!(first_value.is_marked_deleted());
///
/// //All values within this map are individually wrapped in a Mutex implicitly
/// //DataReference derefs to this Mutex, and locking is necessary to actually interact with the value
/// let first_value_lock: MutexGuard<&str> = first_value.lock().unwrap();
/// assert_eq!(*first_value_lock, "Hi");
///```
pub struct DataReference<T, const BLOCK_SIZE: usize> {
	ptr: NonNull<DataHolder<T, BLOCK_SIZE>>,
	phantom: PhantomData<DataHolder<T, BLOCK_SIZE>>,
}

impl<T, const BLOCK_SIZE: usize> DataReference<T, BLOCK_SIZE> {
	fn raw_data(&self) -> &DataHolder<T, BLOCK_SIZE> {
		unsafe { self.ptr.as_ref() }
	}

	/// Has the individual element been marked as deleted?
	///
	/// If this is true, that means this element has been removed from the owning [MbarcMap].
	/// The actual data won't, however, be dropped until all references are gone.
	///
	/// Important to note that this could change at any time, even if a lock is taken on the actual data itself.
	/// You should not rely on this being 100% up-to-date in threaded code, but once this function returns true, it will not become false again.
	pub fn is_marked_deleted(&self) -> bool {
		self.raw_data().deleted()
	}

	/// Get a count of references to the pointed-to data.
	///
	/// It is possible that this number can change even in the time it takes for this function to return.
	/// You should not rely on this being 100% up-to-date in threaded code.
	pub fn ref_count(&self) -> usize {
		self.raw_data().ref_count()
	}

	fn increment_refcount(&self) {
		let inner = self.raw_data();

		let old_rc = inner.ref_count.fetch_add(1, Ordering::Relaxed);
		if old_rc >= isize::MAX as usize {
			std::process::abort();
		}
	}
}

unsafe impl<T: Sync + Send, const BLOCK_SIZE: usize> Send for DataReference<T, BLOCK_SIZE> {}

unsafe impl<T: Sync + Send, const BLOCK_SIZE: usize> Sync for DataReference<T, BLOCK_SIZE> {}

impl<T, const BLOCK_SIZE: usize> Deref for DataReference<T, BLOCK_SIZE> {
	type Target = Mutex<T>;

	fn deref(&self) -> &Self::Target {
		&self.raw_data().data
	}
}

impl<T, const BLOCK_SIZE: usize> Clone for DataReference<T, BLOCK_SIZE> {
	fn clone(&self) -> Self {
		self.increment_refcount();

		Self {
			ptr: self.ptr,
			phantom: PhantomData,
		}
	}
}

impl<T, const BLOCK_SIZE: usize> Drop for DataReference<T, BLOCK_SIZE> {
	fn drop(&mut self) {
		let inner = self.raw_data();

		if inner.ref_count.fetch_sub(1, Ordering::Release) != 1 {
			return;
		}

		atomic::fence(Ordering::Acquire);

		inner.owner.lock().unwrap().remove(&inner.owning_key);
	}
}

/// The heart of this crate, a map which is safe to use in threading contexts without wrapping in a Mutex.
///
/// This map does not hold any locks beyond the duration of its member functions, it is completely safe to hold the results of get indefinitely, regardless of what happens in other threads, and this map will not cause deadlocks
/// This means it is safe to:
/// - Hold multiple [DataReference]s to the same value across multiple threads
/// - Remove elements while a [DataReference] to that value is held elsewhere
/// - Drop the Map itself while [DataReference]s exist elsewhere
/// - Iterate over elements in one thread without taking a lock to the map (iterators hold a reference to each element at time of creation)
///
/// Additionally:
/// - Values are stored in mostly-continuous blocks of memory, and they remain in a fixed address until dropped, making it safe to take references/pointers.
/// - Values are implicitly wrapped in their own Mutex, requiring only that locks are taken on a per-element basis for data access
/// - Values are only dropped when all [DataReference]s to them have been dropped
///
/// This map is not quite a true HashMap, implementing many non-constant (though still sub-linear) operations for managing metadata.
/// The theory behind this approach, however, is that by keeping mutex lock duration to a minimum and everything else "fast enough", any potential performance losses elsewhere should be more than made up accounted for in practice by allowing more saturated thread usage.
///
/// You can kind of think of `MbarcMap<T,U>` as a shorthand for: `Mutex<HashMap<T,Arc<Mutex<U>>>>`, however there's more to it than that, especially in regard to pointer safety (stored values are never moved), memory layout (data is stored in continuous blocks), and iterators (safe to alter the map while iterating over it)
/// MbarcMap may also optionally be defined via a 3rd parameter, BLOCK_SIZE, which determines the size of consecutive blocks in internal storage.  The default for this constant is 32.

pub struct MbarcMap<T: Hash + Eq, U, const BLOCK_SIZE: usize = 32> {
	data: SharedDataContainerType<U, BLOCK_SIZE>,
	data_refs: Arc<Mutex<HashType<T, DataReference<U, BLOCK_SIZE>>>>,
}

impl<T: Hash + Eq, U, const BLOCK_SIZE: usize> MbarcMap<T, U, BLOCK_SIZE> {
	/// Create a new, empty MbarcMap
	///
	/// # Example
	/// ```
	/// use std::backtrace::Backtrace;
	/// use std::sync::Arc;
	/// use mbarc_map::MbarcMap;
	///
	/// let concurrent_map = Arc::new(MbarcMap::<u64,String>::new());
	/// ```
	pub fn new() -> Self {
		Self {
			data: Arc::new(Mutex::new(FaVec::new())),
			data_refs: Arc::new(Mutex::new(HashType::new())),
		}
	}

	/// Inserts a key-value pair into the map.
	///
	/// The return of this function is identical to the insert function of the underlying map type used internally to store references.
	/// Currently, this is std::collections::HashMap
	pub fn insert(&self, key: T, value: U) -> Option<DataReference<U, BLOCK_SIZE>> {
		let mut refs_lock = self.data_refs.lock().unwrap();
		let mut data_lock = self.data.lock().unwrap();

		let new_holder = DataHolder {
			ref_count: AtomicUsize::new(1),
			pending_removal: AtomicBool::new(false),
			data: Mutex::new(value),
			owner: self.data.clone(),
			owning_key: 0,
		};

		let new_key = data_lock.push(new_holder);
		let inserted_item = data_lock.get_mut(&new_key).unwrap();
		inserted_item.owning_key = new_key;

		refs_lock.insert(key, DataReference {
			ptr: NonNull::new(inserted_item as *mut DataHolder<U, BLOCK_SIZE>).unwrap(),
			phantom: PhantomData,
		})
	}

	/// Returns `true` if the map contains a value for the specified key.
	///
	/// Note that, in threaded contexts, this is only correct at the moment this function is called.
	/// It is possible that another thread could add or remove the requested key before you are able to use the return value.
	///
	/// If you intend to use the pattern "if contains then get", then using get alone is sufficient
	pub fn contains(&self, key: &T) -> bool {
		self.data_refs.lock().unwrap().contains_key(key)
	}

	/// Returns a [DataReference] to the value corresponding to the key.  If the key is not present, None will be returned
	///
	/// Note that, in threaded contexts, it is possible for another thread to potentially remove the value you get before you can use it.
	/// In cases like this, the value referenced by the returned [DataReference] will not be dropped until all remaining [DataReference] have been dropped
	pub fn get(&self, key: &T) -> Option<DataReference<U, BLOCK_SIZE>> {
		match self.data_refs.lock().unwrap().get(key) {
			Some(rval) => Some(rval.clone()),
			None => None
		}
	}

	/// Returns a [DataReference] to the value corresponding to the key and removes the key/value from this map.  If the key is not present, None will be returned
	///
	/// The value referenced by the returned [DataReference] will not be dropped until all remaining [DataReference] have been dropped
	pub fn remove(&self, key: &T) -> Option<DataReference<U, BLOCK_SIZE>> {
		match self.data_refs.lock().unwrap().remove(key) {
			Some(value_ref) => {
				value_ref.raw_data().set_deleted(true);
				Some(value_ref)
			}
			None => None
		}
	}

	/// Returns the number of elements in the map.
	pub fn len(&self) -> usize {
		self.data_refs.lock().unwrap().len()
	}

	/// An iterator visiting all values in arbitrary order
	///
	/// Important concurrency note: This iterator will represent the state of the map at creation time.
	/// Adding or removing elements during iteration (in this thread or others) will not have any impact on iteration order, and creation of this iterator has a cost.
	//TODO: make all iterators generated from macro, to ensure they really are "identical"
	pub fn iter(&self) -> crate::minimally_blocking_atomic_reference_counted_map::Iter<DataReference<U, BLOCK_SIZE>> {
		let ref_lock = self.data_refs.lock().unwrap();
		let mut vals = VecDeque::with_capacity(ref_lock.len());

		for value in ref_lock.values() {
			vals.push_back(value.clone());
		}

		Iter {
			items: vals
		}
	}

	/// Exclusive lock on the map for iteration.  This does not clone any elements, therefore requiring a lock is taken on the map.
	/// This returns a [LockedContainer], which only provides an iter() function, returning the iterator you want.
	///
	/// Usage: given some `my_hash: MbarcMap<T,U>`, lock and iterate over it via
	///
	/// `for (k,v) in my_hash.iter_exclusive().iter()`
	pub fn iter_exclusive(&self) -> LockedContainer<'_, T, U, BLOCK_SIZE> {
		LockedContainer {
			items: self.data_refs.lock().unwrap(),
		}
	}
}

impl<T: Hash + Eq + Clone, U, const BLOCK_SIZE: usize> MbarcMap<T, U, BLOCK_SIZE> {
	/// An iterator visiting all key-value pairs in arbitrary order, only for keys which implement Clone
	///
	/// Comments regarding concurrency and performance are the same as in [MbarcMap::iter]
	pub fn iter_cloned_keys(&self) -> Iter<(T, DataReference<U, BLOCK_SIZE>)> {
		let ref_lock = self.data_refs.lock().unwrap();
		let mut vals = VecDeque::with_capacity(ref_lock.len());

		for (key, value) in ref_lock.iter() {
			vals.push_back((key.clone(), value.clone()));
		}

		Iter {
			items: vals
		}
	}
}

impl<T: Hash + Eq + Copy, U, const BLOCK_SIZE: usize> MbarcMap<T, U, BLOCK_SIZE> {
	/// An iterator visiting all key-value pairs in arbitrary order, only for keys which implement Copy
	///
	/// Comments regarding concurrency and performance are the same as in [MbarcMap::iter]
	pub fn iter_copied_keys(&self) -> Iter<(T, DataReference<U, BLOCK_SIZE>)> {
		let ref_lock = self.data_refs.lock().unwrap();
		let mut vals = VecDeque::with_capacity(ref_lock.len());

		for (key, value) in ref_lock.iter() {
			vals.push_back((*key, value.clone()));
		}

		Iter {
			items: vals
		}
	}
}

impl<T: Hash + Eq, U, const BLOCK_SIZE: usize> Drop for MbarcMap<T, U, BLOCK_SIZE> {
	fn drop(&mut self) {
		let ref_lock = self.data_refs.lock().unwrap();

		for value in ref_lock.values() {
			value.raw_data().set_deleted(true);
		}
	}
}

impl<K, V, const BLOCK_SIZE: usize, const N: usize> From<[(K, V); N]> for MbarcMap<K, V, BLOCK_SIZE> where K: Eq + Hash {
	fn from(arr: [(K, V); N]) -> Self {
		let map = Self::new();

		for (k, v) in arr {
			map.insert(k, v);
		}

		map
	}
}

/// An iterator over the entries of a `MbarcMap`.
///
/// This `struct` is created by the various [`iter`] methods on [`MbarcMap`]. See its
/// documentation for more.
///
/// [`iter`]: MbarcMap::iter
pub struct Iter<U> {
	items: VecDeque<U>,
}

impl<U> Iterator for Iter<U> {
	type Item = U;

	fn next(&mut self) -> Option<Self::Item> {
		self.items.pop_front()
	}
}

/// Represents a lock to an [MbarcMap]'s internal map, used exclusively for locked, by-reference iteration
/// see [`iter_exclusive`]
///
/// [`iter_exclusive`]: MbarcMap::iter_exclusive
pub struct LockedContainer<'a, T, U, const BLOCK_SIZE: usize> {
	items: MutexGuard<'a, HashType<T, DataReference<U, BLOCK_SIZE>>>,
}

impl<'a, T, U, const BLOCK_SIZE: usize> LockedContainer<'a, T, U, BLOCK_SIZE> {
	pub fn iter(&self) -> impl Iterator<Item=(&T, &DataReference<U, BLOCK_SIZE>)> {
		self.items.iter()
	}
}
