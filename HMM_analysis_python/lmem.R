library('dplyr')
library('lme4')
library('lmerTest')
library('emmeans')

data <- read.csv("dimensions.csv")
df <- do.call(rbind, lapply(split(data, data$theta), function(dsub) {
		# Make a LMEM of this current subset of the data.
		sample_type <- factor(dsub$sample_type)
		sample_id <- factor(dsub$sample_id)
		dim_mi <- (dsub$dims - dsub$dims_rsm) / (dsub$dims + dsub$dims_rsm)
		model <- lmer(dim_mi ~ sample_type + (1 | sample_id))
		# Calculate the significance scores and add the theta column.
		ret <- summary(emmeans(model, pairwise ~ sample_type)$contrasts)
		ret$theta <- dsub$theta[[1]]
		ret
}))
write.csv(df, "pvalues.csv", row.names=FALSE)
