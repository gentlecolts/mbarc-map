use std::{
	marker::PhantomData,
	ops::Deref,
	ptr::NonNull,
	sync::{
		atomic::{self, Ordering},
		Mutex,
	},
};

use crate::data_holder::DataHolder;
use crate::MbarcMap;

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
/// let concurrent_map = Arc::new(MbarcMap::new());
///
/// let key: i64 = 2;
/// let value: &str = "Hi";
/// concurrent_map.insert(key, value);
///
/// //Retrieve the item we just put into the map, then drop the map itself
/// let first_value: Option<DataReference<&str>> = concurrent_map.get(&key);
/// drop(concurrent_map);
///
/// //Actual data reference
/// let first_value: DataReference<&str> = first_value.unwrap();
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
pub struct DataReference<T> {
	pub(crate) ptr: NonNull<DataHolder<T>>,
	pub(crate) phantom: PhantomData<DataHolder<T>>,
}

impl<T> DataReference<T> {
	pub(crate) fn raw_data(&self) -> &DataHolder<T> {
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

	pub(crate) fn increment_refcount(&self) {
		self.raw_data().increment_refcount();
	}

	pub(crate) fn drop_impl(raw_ptr: NonNull<u8>) {
		let inner = unsafe { raw_ptr.cast::<DataHolder<T>>().as_ref() };

		if inner.ref_count.fetch_sub(1, Ordering::Release) != 1 {
			return;
		}

		atomic::fence(Ordering::Acquire);

		inner.owner.lock().unwrap().remove(&inner.owning_key);
	}
}

unsafe impl<T: Sync + Send> Send for DataReference<T> {}
unsafe impl<T: Sync + Send> Sync for DataReference<T> {}

impl<T> Deref for DataReference<T> {
	type Target = Mutex<T>;

	fn deref(&self) -> &Self::Target {
		&self.raw_data().data
	}
}

impl<T> DataHolder<T> {
	pub(crate) fn make_new_ref(&self) -> Option<DataReference<T>> {
		/*
		Consider the case where, after this function was called, but before this instruction, the last DataReference for this DataHolder had gone out of scope
		when dropped, the DataReference drops count to zero, and thus invalidates this object.
		in practice, this can only happen if a DataReference outside MbarcMap is the last one dropped, and it happens during "ordered" values iteration, which call this function directly
		if the ref count was already zero while this was still being called, then this is "removed", and may be at risk of dropping, and we should not return
		this, however, is incomplete, as it is still possible that both the decrement and drop have completed in a separate thread, leaving &self as an invalid pointer, having been replaced by None within the DataBlock
		while this state does seem "unlikely", and is avoidable it's still realistic, and needs addressing
		it also seems very challenging to create a test case which validates this behavior

		This leaves a few options:
		a) remove the ordered iteration entirely.  benchmarks will be necessary to determine if this is worth consideration
		b) perhaps some combination of MaybeUninit and/or manual drop could resolve this, rely on refcount instead of Option for determining slot availability in DataBlock, etc
		...

		There may be some lock-based solutions to this problem as well, however that is the opposite of the direction we want to head
		ideally, this project could become nearly or completely lock-free, and so this is a problem that needs resolution
		*/
		if self.increment_refcount() == 0 {
			//	self.ref_count.fetch_sub(1, Ordering::Release);
			//	return None;
		}

		Some(DataReference {
			ptr: NonNull::from(self),
			phantom: PhantomData,
		})
	}
}

impl<T> Clone for DataReference<T> {
	fn clone(&self) -> Self {
		//since we already are a DataReference, we must have at least one ref, this cannot fail
		self.raw_data().make_new_ref().unwrap()
	}
}

impl<T> Drop for DataReference<T> {
	fn drop(&mut self) {
		DataReference::<T>::drop_impl(self.ptr.cast::<u8>());
	}
}
