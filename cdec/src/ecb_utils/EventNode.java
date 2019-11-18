package ecb_utils;

import java.io.File;
import java.util.Objects;

public class EventNode {
	
	public final File file;
	public final String m_id, ev_id;

	public EventNode(File file, String m_id, String ev_id) {
		this.file = Objects.requireNonNull(file, "file");
		this.m_id = Objects.requireNonNull(m_id, "m_id");
		this.ev_id = Objects.requireNonNull(ev_id, "ev_id");
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof EventNode) {
			EventNode otherEv = (EventNode) obj;
			boolean eq = this.file.equals(otherEv.file) 
							&& this.m_id.equals(otherEv.m_id) 
							&& this.ev_id.equals(otherEv.ev_id);
			return eq;
		}
		return false;
	}
	
	public String getGlobalKey() {
		return this.file.getName() + "_" + this.m_id;	
	}
	
	public boolean corefers(EventNode other) {
		return this.ev_id.equals(other.ev_id);
	}
	
	public String getTopic() {
		return this.file.getName().split("_")[0];
	}
	public String getSubTopic() {
		String sub = "";
		if(this.file.getName().contains("ecbplus"))
			sub = "ecbplus";
		else
			sub = "ecb";
		return sub;
	}
	@Override
	public int hashCode() {
		return Objects.hash(this.file, this.m_id);
	}

	@Override
	public String toString() {
		return "(" + this.file.getName() + ", " + this.m_id + ", " + this.ev_id + ")";
	}

}
