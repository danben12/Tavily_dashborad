Tavily Data Analyst - Home Assignment

Presented and analyzed by: Dan Benbenisti

Note: the text is AI edited, but the ideas and insights are authentic and genuine.

Assumption: The dataset provided reflects Research API users only. As a partial dataset, it cannot support conclusions regarding broader product usage, costs, or efficiency.

Part I: Product Analysis

Q1: Does the Research API successfully retain new users, and what is the profile of the users driving its traffic?

The Research API accounts for a significant 19% of first-time new user interactions since Nov 1st, 2025. Given this high acquisition volume, it's worth exploring if the API keeps these users engaged over time, or if they mostly just use it once. Furthermore, analyzing traffic distribution is essential to understand the nature of the product and the specific audience it serves.

Hypothesis: The Research API will successfully attract new users who will remain engaged with the platform over the long term.

Key Finding: About 21.4% of users who started with the Research API never performed another action and just abandoned the platform. This immediate abandonment rate is significantly higher than any other feature on the platform. A deeper analysis of the usage patterns reveals that most of the activity within this feature is driven by a small segment of users as 5% of Research API users are responsible for approximately 60% of all requests within this feature.

Conclusion: The data suggests the Research API functions as a specialized "power tool" rather than a mainstream feature. High initial interest indicates general curiosity, but factors like credit consumption, latency, and workflow complexity likely cause many to try the API once and never return to the platform. This leaves a small, dedicated niche of power users who drive most of the traffic, since the tool successfully addresses their specific recurring need. This means the Research API shouldn't be judged by general retention standards, but rather as a high-value tool for a select group of heavy users.

Recommendations:

Introduce Alternative Features: To prevent most users from abandoning the platform entirely after trying the API, they should be proactively exposed to standard, more accessible features immediately afterward. This helps ensure they are retained as platform users even if the heavy research tool isn't for them.

Offer Subscriptions for Heavy Users: To guarantee the long-term retention of heavy users, they should be offered a dedicated, discounted monthly subscription to secure their loyalty while generating stable, predictable revenue for the company.

Q2: Is the current pricing model viable, and is the Research API economically sustainable?

While the Research API appears to help attract new users, a critical question remains: at what cost? Evaluating the economic impact of operating the Research API helps assess its financial viability and sustainability. For the API to remain available and scalable in the long term, its underlying pricing model must be profitable. Therefore, it is important to understand whether the product is currently operating at a loss, and if so, where and why these losses occur.

Hypothesis: The Research API is a profitable tool and a sustainable revenue driver that will contribute to the company's long-term financial growth.

Key Finding: 92.1% of all users belong to the strictly free group (meaning they have no paying subscription and no pay-as-you-go plan), generating zero revenue for the company. The core issue is that these free users are currently granted unrestricted access to the heavy "Pro" model, which costs the company an average of $134 per request, compared to just $36 for the "Mini" model. Fully free users took advantage of this open access to make 33K requests using the expensive Pro model. As a result, the company absorbed a massive direct cost of approximately $4.42M to support non-paying users, many of whom leave the platform after first activity.

Conclusion: The current free tier setup is economically unsustainable. Providing unrestricted access to premium, high-cost models without generating any revenue turns a simple marketing tool into a huge financial loss.

Recommendations:

Restrict Pro Model Access and Default to Mini Model: Access to the Pro model should be restricted to paying users, with all free-tier users routed to the Mini model by default. If the requests made by free users were processed using the Mini model (at ~$36.20 per request), the total cost would have dropped to roughly $1.19M, saving the company approximately $3.22M over a similar timeframe.

Audit Pricing: Alongside curbing free user costs, an audit is required to ensure that the rate charged to paying customers covers the true, high cost of running the Pro model (~$134).

Data Note: It is understood and has been advised that the values in the "request cost" column of the research requests dataset are denominated in full US Dollars (USD), not cents.

Q3: What causes users to cancel requests mid-stream, and what are the effects of these cancellations?

A mid-run cancellation of a research request exposes a combined failure in user experience and the billing model. The system burns expensive infrastructure resources, only for the user to abort the operation. Understanding the users' breaking point and fixing the billing mechanism is important to stop unnecessary resource drain on one hand and prevent the loss of customer trust on the other.

Hypothesis: Mid-stream request cancellations are primarily driven by human user when wait times exceed a tolerable threshold

Key Finding: An analysis of 1,840 canceled requests reveals that 100% of these cancellations occurred when real-time streaming was enabled, under the assumption that streaming is used exclusively by human users. In contrast, automated agents making approximately 100K non-streaming calls recorded no cancellations. This indicates that cancellations are driven entirely by human users, likely due to a combination of human errors in the initial prompt or a loss of patience. The data identifies a clear breaking point in research API streaming request running times: for requests completing in under 90 seconds, the cancellation rate is a relatively low 6%. However, once the wait time crosses the 90-second mark, cancellations spike to 24.1%. These slow requests are less efficient, often using a median of 24 LLM calls while yielding the same median number of sources (19). Cancellations typically occur after a median of 92 seconds, as the system fails to provide a conclusive response despite the extensive background processing. Furthermore, a billing failure exists - in 98.7% of cancellation cases, the user was not charged any credits, leaving the company to absorb the costs (averaging ~$6.19 per request). Conversely, the remaining 1.3% of users were charged credits but received no final output, which negatively impacts the user experience.

Conclusion: Long wait times might cause user frustration for human users, leading to high cancellation rates. Adding to this is a flawed billing mechanism that forces the company to absorb the costs of these abandoned requests.

Recommendations:

Implement a Timeout: An automatic stop should be enforced for streaming requests at the ~80-second mark. At this point, the system should stop and display the answers and sources gathered up to that moment. This provides immediate value for the user, prevents abandonment, and allows for legitimate billing.

Refine the Billing Policy: If timeout policy is not implemented, or to address manual cancellations occurring before the mark, the billing mechanism should be adjusted to reflect the actual compute resources consumed. Users should be billed for the searches conducted up to the moment of cancellation, while ensuring they retain access to all partial results generated.

Part II: Infrastructure & Cost Analysis

Key Questions: What components drive the bulk of our operating costs? How efficiently does our infrastructure adapt and scale relative to actual user demand?

Key Metrics evaluated: Total cost, Hourly Infrastructure Cost, Volume-to-Cost Correlation, and Inactive Resource Cost.

The analysis reveals a clear structural split: hardware infrastructure is the primary cost driver, accounting for ~95.6% of total spending, while all model costs combined amount to just 4.4%. At a high level, hourly infrastructure costs appear stable, fluctuating between $82–$138/hr with a median of ~$105/hr. A daily resolution analysis reveals a meaningful correlation between daily request volume and daily total cost (Pearson r = 0.88), indicating that costs are, to some degree, driven by request volume.

When cross-referencing infrastructure spend against request volume and user growth between November 2025 and March 2026, a notable disconnect is observed. New user registrations grew X14 and monthly request volume surged X4.3 , yet infrastructure costs rose only X1.1 over the same period - likely because the system was built to handle peak load from day one. Consequently, the company effectively pre-paid for capacity it only recently began to use, leading to significant capital inefficiency during the early months.

A closer look at activity and costs distribution in the hour and day of the week resolution, reveals a consistent pattern: costs peak during daytime hours on weekdays and drop during nights and weekends. Yet, even during these low-activity windows, the infrastructure barely adjusts. On weekends, overall traffic drops by 28.5%, yet infrastructure costs decline by ~3%, with servers remaining fully active through low-demand periods with no dynamic adjustment. This is most evident in the Research cluster: as established in the product analysis segment, the Research API is a product targeted at a specific user base, which naturally results in lower and more sporadic activity. Of 3,611 hours analyzed, 32% recorded zero research requests, yet the cluster continued running at full capacity - burning an estimated $14,235 on zero-traffic hours.

Data note: A discrepancy exists between the data in product dataset and the total infrastructure costs. The gap can be explained, for example by paid access to external data sources etc.

Recommendations:

Optimize baseline capacity: Given that the product is still in a growth phase, the current resource allocation is too rigid. Rather than committing resources for peak capacity upfront, the infrastructure baseline should be reduced and scaled up incrementally as demand grows over time to eliminate unnecessary expenses.

Shut down the Research cluster during inactivity: Given the high share of inactive hours on the Research cluster, it should be configured to automatically shut down after ~30 minutes of inactivity and restart when the next request arrives. This alone could yield meaningful cost savings without impacting user experience.

Part III: Next Steps & Further Investigation

Power User Profiling: Analyze the behavior, use cases, and payment plan of the top 5% of users driving traffic to deeper understand the profile of the users.

Pricing Model Simulation & Testing: Develop a simulation model to project the cost savings and revenue due to optimization

Predictive Resource Provisioning: Analyze historical daily and hourly traffic patterns to develop a dynamic scaling model.
