use std::sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, Arc, Mutex};

use crate::fixed_address_continuous_allocation::{FaVec, FaVecIndex};

pub(crate) type SharedDataContainerType<T> = Arc<Mutex<FaVec<DataHolder<T>, 32>>>;


pub(crate) struct DataHolder<T> {
	pub(crate) ref_count: AtomicUsize,
	pub(crate) pending_removal: AtomicBool,

	//TODO: having these here rather than in DataReference means less duplication, but also makes for fatter entries in the FaVec array
	//the deduplication is probably worth it, but will need to bench for it being worth the potentially less good caching behavior
	pub(crate) owner: SharedDataContainerType<T>,
	pub(crate) owning_key: FaVecIndex<32>,

	pub data: Mutex<T>,
}

impl<T> DataHolder<T> {
	pub(crate) fn deleted(&self) -> bool {
		//TODO: evaluate safety of this ordering
		self.pending_removal.load(Ordering::Acquire)
	}

	pub(crate) fn ref_count(&self) -> usize {
		//TODO: evaluate safety of this ordering
		self.ref_count.load(Ordering::Acquire)
	}

	pub(crate) fn set_deleted(&self) {
		//TODO: evaluate safety of this ordering
		self.pending_removal.store(true, Ordering::Release);
	}
}