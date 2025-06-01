use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::data_holder::{DataHolder, SharedDataContainerType};
use crate::data_reference::DataReference;
use crate::fixed_address_continuous_allocation::{FaVec, FaVecIndex};

type HashType<T, U> = HashMap<T, U>;

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
/// This map is not quite a true HashMap, implementing many non-constant (though still sublinear) operations for managing metadata.
/// The theory behind this approach, however, is that by keeping mutex lock duration to a minimum and everything else "fast enough", any potential performance losses elsewhere should be more than made up accounted for in practice by allowing more saturated thread usage.
pub struct MbarcMap<T: Hash + Eq, U> {
	data: SharedDataContainerType<U>,
	data_refs: Arc<Mutex<HashType<T, DataReference<U>>>>,
}

impl<T: Hash + Eq, U> Default for MbarcMap<T, U> {
	fn default() -> Self {
		Self::new()
	}
}

impl<T: Hash + Eq, U> MbarcMap<T, U> {
	/// Create a new, empty MbarcMap
	///
	/// # Example
	/// ```
	/// use std::sync::Arc;
	/// use mbarc_map::MbarcMap;
	///
	/// let concurrent_map = Arc::new(MbarcMap::<u64,String>::new());
	/// ```
	pub fn new() -> Self {
		Self {
			data: Arc::new(FaVec::new()),
			data_refs: Arc::new(Mutex::new(HashType::new())),
		}
	}

	/// Inserts a key-value pair into the map.
	///
	/// The return of this function is identical to the insert function of the underlying map type used internally to store references.
	/// Currently, this is std::collections::HashMap
	pub fn insert(&self, key: T, value: U) -> Option<DataReference<U>> {
		let new_holder = DataHolder {
			ref_count: AtomicUsize::new(1),
			pending_removal: AtomicBool::new(false),
			data: Mutex::new(value),
			owner: self.data.clone(),
			owning_key: FaVecIndex::from_absolute_index(0),
		};

		let new_key = self.data.push(new_holder);

		unsafe {
			let inserted_item = self.data.get_raw(&new_key).unwrap();
			(*inserted_item).owning_key = new_key;

			self.data_refs.lock().unwrap().insert(
				key,
				DataReference {
					ptr: NonNull::new(inserted_item).unwrap(),
					phantom: PhantomData,
				},
			)
		}
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
	pub fn get(&self, key: &T) -> Option<DataReference<U>> {
		self.data_refs.lock().unwrap().get(key).cloned()
	}

	/// Returns a [DataReference] to the value corresponding to the key and removes the key/value from this map.  If the key is not present, None will be returned
	///
	/// The value referenced by the returned [DataReference] will not be dropped until all remaining [DataReference] have been dropped
	pub fn remove(&self, key: &T) -> Option<DataReference<U>> {
		match self.data_refs.lock().unwrap().remove(key) {
			Some(value_ref) => {
				value_ref.raw_data().set_deleted();
				Some(value_ref)
			}
			None => None,
		}
	}

	/// Returns the number of elements in the map.
	pub fn len(&self) -> usize {
		self.data_refs.lock().unwrap().len()
	}

	///Returns `true` if the map contains no elements.
	pub fn is_empty(&self) -> bool {
		self.data_refs.lock().unwrap().is_empty()
	}

	/// An iterator visiting all values in arbitrary order
	///
	/// Important concurrency note: This iterator will represent the state of the map at creation time.
	/// Adding or removing elements during iteration (in this thread or others) will not have any impact on iteration order, and creation of this iterator has a cost.
	//TODO: make all iterators generated from macro, to ensure they really are "identical" (????)
	pub fn iter(
		&self,
	) -> crate::minimally_blocking_atomic_reference_counted_map::Iter<DataReference<U>> {
		let ref_lock = self.data_refs.lock().unwrap();
		let mut vals = VecDeque::with_capacity(ref_lock.len());

		for value in ref_lock.values() {
			vals.push_back(value.clone());
		}

		Iter { items: vals }
	}

	/// Value only iterator representing the values in this map at the time it is called.  The order of iteration will be the in-memory order of values, and this should be preferred if keys are not needed
	pub fn iter_copied_values_ordered(&self) -> std::vec::IntoIter<DataReference<U>> {
		let mut items = self.iter().collect::<Vec<_>>();
		items.sort_by_cached_key(|i| i.raw_data().owning_key.as_absolute_index());

		items.into_iter()
	}

	/// Exclusive lock on the map for iteration.  This does not clone any element references, therefore requiring a lock is taken on the map.
	/// This returns a [LockedContainer], which only provides an iter() function, returning the iterator you want.
	///
	/// Usage: given some `my_hash: MbarcMap<T,U>`, lock and iterate over it via
	///
	/// `for (k,v) in my_hash.iter_exclusive().iter()`
	pub fn iter_exclusive(&self) -> LockedContainer<'_, T, U> {
		LockedContainer {
			items: self.data_refs.lock().unwrap(),
		}
	}
}

impl<T: Hash + Eq + Clone, U> MbarcMap<T, U> {
	/// An iterator visiting all key-value pairs in arbitrary order, only for keys which implement Clone
	///
	/// Comments regarding concurrency and performance are the same as in [MbarcMap::iter]
	pub fn iter_cloned_keys(&self) -> Iter<(T, DataReference<U>)> {
		let ref_lock = self.data_refs.lock().unwrap();
		let mut vals = VecDeque::with_capacity(ref_lock.len());

		for (key, value) in ref_lock.iter() {
			vals.push_back((key.clone(), value.clone()));
		}

		Iter { items: vals }
	}
}

impl<T: Hash + Eq + Copy, U> MbarcMap<T, U> {
	/// An iterator visiting all key-value pairs in arbitrary order, only for keys which implement Copy
	///
	/// Comments regarding concurrency and performance are the same as in [MbarcMap::iter]
	pub fn iter_copied_keys(&self) -> Iter<(T, DataReference<U>)> {
		let ref_lock = self.data_refs.lock().unwrap();
		let mut vals = VecDeque::with_capacity(ref_lock.len());

		for (key, value) in ref_lock.iter() {
			vals.push_back((*key, value.clone()));
		}

		Iter { items: vals }
	}
}

unsafe impl<T: Hash + Eq, U> Send for MbarcMap<T, U> {}
unsafe impl<T: Hash + Eq, U> Sync for MbarcMap<T, U> {}

impl<T: Hash + Eq, U> Drop for MbarcMap<T, U> {
	fn drop(&mut self) {
		let ref_lock = self.data_refs.lock().unwrap();

		for value in ref_lock.values() {
			value.raw_data().set_deleted();
		}
	}
}

impl<K, V, const N: usize> From<[(K, V); N]> for MbarcMap<K, V>
where
	K: Eq + Hash,
{
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
pub struct LockedContainer<'a, T, U> {
	items: MutexGuard<'a, HashType<T, DataReference<U>>>,
}

impl<'a, T, U> LockedContainer<'a, T, U> {
	pub fn iter(&self) -> impl Iterator<Item = (&T, &DataReference<U>)> {
		self.items.iter()
	}
}
