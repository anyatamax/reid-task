import torchvision
import numpy as np
from PIL import Image

def log_similar_texts_and_images_with_dynamic_graph(
    lightning_module, batch_img, batch_target, current_epoch, global_step, n_similar=5
):
    """
    Logging function that builds TextGraphSampler dynamically when graph_sampling is not used.
    """
    try:
        if not hasattr(lightning_module.trainer, 'datamodule'):
            print("No datamodule found in trainer")
            return
            
        datamodule = lightning_module.trainer.datamodule
        train_loader = lightning_module.trainer.train_dataloader
        
        # Import TextGraphSampler
        from datasets.graph_sampler import TextGraphSampler
        
        # Get captions from train_set
        captions_map = getattr(datamodule.train_set, 'captions', {})
        
        if not captions_map:
            print("⚠️ No captions available - cannot build text-based similarity graph")
            return
        
        # Build TextGraphSampler dynamically
        print("🔨 Building TextGraphSampler dynamically for logging...")
        graph_sampler = TextGraphSampler(
            data_source=datamodule.dataset.train,
            model=lightning_module.model,
            captions_map=captions_map,
            batch_size=datamodule.batch_size_stage2,
            num_instance=datamodule.num_instance,
            verbose=True
        )
        
        # Build similarity graph
        print("📊 Building similarity graph...")
        graph_sampler.make_index()
        
        if graph_sampler.similarity_graph is None:
            print("❌ Failed to build similarity graph")
            return
        
        print(f"✅ Similarity graph built with shape: {graph_sampler.similarity_graph.shape}")
        
        # Now use the graph for logging
        batch_target_cpu = batch_target.cpu().numpy()
        unique_classes_in_batch = np.unique(batch_target_cpu)
        
        anchor_pid = unique_classes_in_batch[0]
        
        images_for_grid = []
        texts_for_logging = []
        
        # Log anchor class
        anchor_mask = batch_target_cpu == anchor_pid
        anchor_indices = np.where(anchor_mask)[0]
        
        if len(anchor_indices) > 0:
            anchor_img_idx = anchor_indices[0]
            anchor_img_tensor = batch_img[anchor_img_idx]
            images_for_grid.append(anchor_img_tensor.cpu())
            
            texts_for_logging.append(f"ANCHOR Class {anchor_pid}")
            
            anchor_images = graph_sampler.pid_to_paths[anchor_pid]
            caption = graph_sampler._get_caption_for_image(anchor_images[0])
            if caption:
                texts_for_logging.append(f"Caption: {caption[:150]}...")
            else:
                texts_for_logging.append("Caption: Default prompt used")
            
            texts_for_logging.append("")
        
        # Log other classes from batch
        other_classes = [pid for pid in unique_classes_in_batch if pid != anchor_pid]
        
        for i, other_pid in enumerate(other_classes[:n_similar]):
            other_mask = batch_target_cpu == other_pid
            other_indices = np.where(other_mask)[0]
            
            if len(other_indices) > 0:
                other_img_idx = other_indices[0]
                other_img_tensor = batch_img[other_img_idx]
                images_for_grid.append(other_img_tensor.cpu())
                
                texts_for_logging.append(f"BATCH Class {other_pid}")
                
                other_images = graph_sampler.pid_to_paths[other_pid]
                caption = graph_sampler._get_caption_for_image(other_images[0])
                if caption:
                    texts_for_logging.append(f"Caption: {caption[:150]}...")
                else:
                    texts_for_logging.append("Caption: Default prompt used")
                
                texts_for_logging.append("")
        
        # Create grid for batch
        if len(images_for_grid) > 0:
            images_grid = torchvision.utils.make_grid(
                images_for_grid,
                nrow=min(4, len(images_for_grid)),
                padding=15,
                normalize=True,
                pad_value=1.0
            )
            
            lightning_module.logger.experiment.add_image(
                f"training_batch/anchor_{anchor_pid}_batch",
                images_grid,
                global_step=global_step
            )
            
            text_summary = "\n".join(texts_for_logging)
            lightning_module.logger.experiment.add_text(
                f"training_batch/anchor_{anchor_pid}_analysis",
                text_summary,
                global_step=global_step
            )
        
        # Log top similar classes from similarity graph
        anchor_pid_idx = graph_sampler.pids.index(anchor_pid) if anchor_pid in graph_sampler.pids else None
        
        if anchor_pid_idx is not None and anchor_pid_idx < len(graph_sampler.similarity_graph):
            similar_indices = graph_sampler.similarity_graph[anchor_pid_idx][:n_similar]
            similar_pids = [graph_sampler.pids[idx] for idx in similar_indices if idx < len(graph_sampler.pids)]
            
            texts_similar = [f"Top {len(similar_pids)} most similar classes to anchor {anchor_pid} (from similarity graph):", ""]
            images_similar = []
            
            for rank, similar_pid in enumerate(similar_pids, 1):
                if similar_pid in graph_sampler.pid_to_paths:
                    images = graph_sampler.pid_to_paths[similar_pid]
                    if images:
                        try:
                            img_path = images[0]
                            img = Image.open(img_path).convert('RGB')
                            img_tensor = datamodule.train_transforms(img)
                            images_similar.append(img_tensor.cpu())
                            
                            texts_similar.append(f"Rank {rank}: Class {similar_pid}")
                            caption = graph_sampler._get_caption_for_image(img_path)
                            if caption:
                                texts_similar.append(f"Caption: {caption[:150]}...")
                            else:
                                texts_similar.append("Caption: Default prompt used")
                            texts_similar.append("")
                        except Exception as e:
                            print(f"Could not load image for class {similar_pid}: {e}")
            
            if len(images_similar) > 0:
                similar_grid = torchvision.utils.make_grid(
                    images_similar,
                    nrow=min(4, len(images_similar)),
                    padding=15,
                    normalize=True,
                    pad_value=1.0
                )
                
                lightning_module.logger.experiment.add_image(
                    f"training_similar/anchor_{anchor_pid}_top_similar",
                    similar_grid,
                    global_step=global_step
                )
                
                similar_summary = "\n".join(texts_similar)
                lightning_module.logger.experiment.add_text(
                    f"training_similar/anchor_{anchor_pid}_top_similar_analysis",
                    similar_summary,
                    global_step=global_step
                )
        
        print(f"✅ Logged batch analysis for anchor class {anchor_pid} with dynamic graph ({len(images_for_grid)} images in batch)")
        
    except Exception as e:
        print(f"❌ Error in log_similar_texts_and_images_with_dynamic_graph: {e}")
        import traceback
        traceback.print_exc()


def log_similar_texts_and_images(
    lightning_module, batch_img, batch_target, current_epoch, global_step, n_similar=5
):
    try:
        if not hasattr(lightning_module.trainer, 'datamodule'):
            print("No datamodule found in trainer")
            return
            
        train_loader = lightning_module.trainer.train_dataloader

        if not lightning_module.trainer.datamodule.use_graph_sampling:
            print("Graph Sampling (TextGraphSampler) not used")
            return
            
        sampler = train_loader.sampler

        batch_target_cpu = batch_target.cpu().numpy()
        unique_classes_in_batch = np.unique(batch_target_cpu)

        if sampler.similarity_graph is None:
            print("No similarity graph available - skipping analysis")
            return
        
        anchor_pid = unique_classes_in_batch[0]

        images_for_grid = []
        texts_for_logging = []
        
        anchor_mask = batch_target_cpu == anchor_pid
        anchor_indices = np.where(anchor_mask)[0]
        
        if len(anchor_indices) > 0:
            # Берем первое изображение anchor класса из батча
            anchor_img_idx = anchor_indices[0]
            anchor_img_tensor = batch_img[anchor_img_idx]
            images_for_grid.append(anchor_img_tensor.cpu())
            
            texts_for_logging.append(f"ANCHOR Class {anchor_pid}")

            anchor_images = sampler.pid_to_paths[anchor_pid]
            caption = sampler._get_caption_for_image(anchor_images[0])
            if caption:
                texts_for_logging.append(f"Caption: {caption[:150]}...")
            else:
                texts_for_logging.append("Caption: Default prompt used")
            
            texts_for_logging.append("")
        
        # Добавляем остальные классы из батча
        other_classes = [pid for pid in unique_classes_in_batch if pid != anchor_pid]
        
        for i, other_pid in enumerate(other_classes[:n_similar]):
            other_mask = batch_target_cpu == other_pid
            other_indices = np.where(other_mask)[0]
            
            if len(other_indices) > 0:
                other_img_idx = other_indices[0]
                other_img_tensor = batch_img[other_img_idx]
                images_for_grid.append(other_img_tensor.cpu())
                
                texts_for_logging.append(f"BATCH Class {other_pid}")

                other_images = sampler.pid_to_paths[other_pid]
                caption = sampler._get_caption_for_image(other_images[0])
                if caption:
                    texts_for_logging.append(f"Caption: {caption[:150]}...")
                else:
                    texts_for_logging.append("Caption: Default prompt used")
                
                texts_for_logging.append("")
        
        images_grid = torchvision.utils.make_grid(
            images_for_grid, 
            nrow=min(4, len(images_for_grid)), 
            padding=15,
            normalize=True,
            pad_value=1.0
        )

        lightning_module.logger.experiment.add_image(
            f"graph_sampling_batch/anchor_{anchor_pid}_batch", 
            images_grid, 
            global_step=global_step
        )

        text_summary = "\n".join(texts_for_logging)
        lightning_module.logger.experiment.add_text(
            f"graph_sampling_batch/anchor_{anchor_pid}_analysis",
            text_summary,
            global_step=global_step
        )

        print(f"Logged batch analysis for anchor class {anchor_pid} ({len(images_for_grid)} images)")
        
    except Exception as e:
        print(f"❌ Error in log_similar_texts_and_images: {e}")
        import traceback
        traceback.print_exc()