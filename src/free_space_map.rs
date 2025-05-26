use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(crate) struct FreeSpaceMap<const BLOCK_SIZE: usize> {
	//maps remaining free blocks to the (set of) indices of corresponding data blocks
	//TODO: would be nice if we could make map private
	pub(crate) map: BTreeMap<usize, BTreeSet<usize>>,
	tracked_block_count: usize,
}

impl<const BLOCK_SIZE: usize> FreeSpaceMap<BLOCK_SIZE> {
	/// Identifies the best-available data block for inserting a new element, updates the size of that element in this map
	/// if there are no free blocks, this will store current_data_block_count in the map and return it, indicating the need for a new block
	/// returns (index of data block, current free space of block)
	pub(crate) fn get_most_suitable_block_index_for_push(&mut self) -> (usize, usize) {
		//the current strategy here is to first, find the block(s) with the least amount of non-zero free space remaining
		//we want to concentrate data to "hot" blocks, where there's already lots of other data, so we have fewer gaps
		//from these blocks, we then select the lowest-index block, so that high index blocks are more likely to empty, and thus be dropped later during remove()

		let lowest_free_space = self.map.keys().find(|key| **key > 0);

		//if no blocks with more than 0 free space, need to make a new block
		let lowest_free_space = match lowest_free_space {
			Some(lowest_free_space) => {
				//println!("lowest has {} free blocks", lowest_free_space);
				*lowest_free_space
			}
			None => {
				//add new data block
				let new_index = self.tracked_block_count;
				self.tracked_block_count += 1;

				//track the new block's free space in our map
				let new_set = BTreeSet::from([new_index]);
				self.map.insert(BLOCK_SIZE, new_set);

				//we now have a block that's completely empty
				BLOCK_SIZE
			}
		};

		//pick the lowest-indexed block (with the least free space)
		//println!("{:?}", self.free_space_map);
		let possible_blocks = self.map.get_mut(&lowest_free_space).unwrap();
		assert!(!possible_blocks.is_empty());
		let block_index = possible_blocks.first().unwrap();

		//println!("{:?}", self.free_space_map);
		assert!(block_index < &self.tracked_block_count);
		(*block_index, lowest_free_space)
	}

	fn update_block_free_space(
		&mut self,
		block_index: usize,
		old_free_space: usize,
		new_free_space: usize,
	) {
		let possible_blocks = self.map.get_mut(&old_free_space).unwrap();
		assert!(!possible_blocks.is_empty());
		possible_blocks.remove(&block_index);

		//if no more blocks have lowest_free_space left, then remove the index
		if possible_blocks.is_empty() {
			self.map.remove(&old_free_space);
		}

		match self.map.entry(new_free_space) {
			Entry::Vacant(entry) => {
				entry.insert(BTreeSet::from_iter([block_index]));
			}
			Entry::Occupied(entry) => {
				entry.into_mut().insert(block_index);
			}
		}
	}

	//TODO: this function becomes much more efficient if we also track [block index] => [free space]
	//however, care needs to be taken to enable future attempts to make this struct lock-free
	fn find_block_free_space(&self, block_index: usize) -> Option<usize> {
		for (free_space, blocks) in self.map.iter() {
			if blocks.contains(&block_index) {
				return Some(*free_space);
			}
		}

		None
	}

	pub(crate) fn increment_block_free_space(&mut self, block_index: usize) {
		if let Some(free_space) = self.find_block_free_space(block_index) {
			self.update_block_free_space(block_index, free_space, free_space + 1);
		}
	}

	pub(crate) fn decrement_block_free_space(&mut self, block_index: usize) {
		if let Some(free_space) = self.find_block_free_space(block_index) {
			self.update_block_free_space(block_index, free_space, free_space - 1);
		}
	}

	/// Remove the highest index item from this map
	/// this should correspond to the last data block, which should only be popped when empty
	/// therefore, we can assert that we only pop from the set of items with BLOCK_SIZE free space
	//TODO: to safely become lock-free, we may simply never remove allocated blocks, at which point this can be removed
	fn pop(&mut self) -> usize {
		let end_block_index = self.tracked_block_count - 1;

		assert!(self.map.contains_key(&BLOCK_SIZE));
		let free_indices = self.map.get_mut(&BLOCK_SIZE).unwrap();

		assert!(free_indices.contains(&end_block_index));
		free_indices.remove(&end_block_index);

		end_block_index
	}
}
