library('dplyr')
library('lme4')
library('lmerTest')
library('emmeans')

dimensions <- read.csv("dimensions.csv")
dsub = dimensions[dimensions$theta == 0.75, ]
# Make a LMEM of this current subset of the data.
sample_type <- factor(dsub$sample_type)
sample_id <- factor(dsub$sample_id)
model <- glmer(dsub$dims ~ sample_type + (1 | sample_id), family=poisson(link="log"))
emmeans(model, pairwise ~ sample_type)

df <- do.call(rbind, lapply(split(dimensions, dimensions$theta), function(dsub) {
		# Make a LMEM of this current subset of the data.
		sample_type <- factor(dsub$sample_type)
		sample_id <- factor(dsub$sample_id)
		model <- glmer(dsub$dims ~ sample_type + (1 | sample_id), family=poisson(link="log"))
		# Calculate the significance scores and add the theta column.
		ret <- summary(emmeans(model, pairwise ~ sample_type)$contrasts)
		ret$theta <- dsub$theta[[1]]
		ret
}))
write.csv(df, "pvalues.csv", row.names=FALSE)


# Check the state traversal rates
traversal <- read.csv("traversal.csv")
sample_type <- factor(traversal$sample_type)
sample_id <- factor(traversal$sample_id)
K <- factor(traversal$K)
model <- lmer(traversal$rate ~ sample_type + K + (1 | sample_id))
emmeans(model, pairwise ~ sample_type)
